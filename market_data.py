"""Bounded public data requests; no Streamlit or model work in this module."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
import logging
import re
from zoneinfo import ZoneInfo
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from xlrd import XLRDError

LOG = logging.getLogger(__name__)
CN = ZoneInfo("Asia/Shanghai")
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
INDICES = {"000300.SS": "沪深300", "000001.SS": "上证指数",
           "399006.SZ": "创业板指", "000905.SS": "中证500"}
PROVIDERS = ("腾讯", "东方财富")


class DataError(RuntimeError):
    """A user-visible data failure, kept separate from model failures."""


def require_object(value, label):
    if not isinstance(value, dict):
        raise DataError(f"{label}响应格式异常：预期 JSON 对象")
    return value


@dataclass
class History:
    frame: pd.DataFrame
    source: str
    name: str
    fetched_at: str
    adjustment: str
    notes: list[str] = field(default_factory=list)


def now_cn():
    return datetime.now(CN)


def normalize_a_share_code(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    if re.fullmatch(r"\d{1,6}\.0", text):
        text = text[:-2]
    match = re.fullmatch(r"(?:(SH|SZ))?(\d{1,6})(?:\.(SH|SS|SZ))?", text)
    if not match:
        return None
    prefix, code, suffix = match.groups()
    code = code.zfill(6)
    if code.startswith(("60", "68", "50", "51", "52", "56", "58")):
        exchange = "SS"
    elif code.startswith(("00", "30", "15", "16", "18")) and code != "000000":
        exchange = "SZ"
    else:
        return None
    for supplied in (prefix, suffix):
        if supplied and supplied.replace("SH", "SS") != exchange:
            return None
    return f"{code}.{exchange}"


def parse_codes(text):
    valid, invalid = [], []
    for raw in re.split(r"[\s,，;；、]+", text.strip()):
        if not raw:
            continue
        code = normalize_a_share_code(raw)
        if code is None:
            invalid.append(raw)
        elif code not in valid:
            valid.append(code)
    return valid, invalid


def normalize_constituent_frame(frame):
    # Check None before looking at .columns (the original fallback crashed here).
    empty = pd.DataFrame(columns=["代码", "名称"])
    if frame is None or frame.empty:
        return empty
    columns = {str(c).replace(" ", "").lower(): c for c in frame.columns}
    code_col = next((original for key, original in columns.items()
                     if key in ("code", "symbol", "代码", "股票代码", "证券代码", "品种代码")
                     or "constituentcode" in key or "成分券代码" in key or "成份券代码" in key), None)
    name_col = next((original for key, original in columns.items()
                     if key in ("name", "名称", "股票简称", "证券简称", "品种名称")
                     or "constituentname" in key and "eng" not in key
                     or key in ("成分券名称", "成份券名称")), None)
    if code_col is None:
        return empty
    result = pd.DataFrame({"代码": frame[code_col].map(normalize_a_share_code)})
    result["名称"] = frame[name_col].fillna("").astype(str) if name_col is not None else ""
    result = result.dropna(subset=["代码"])
    # A constituent list must contain equities, not funds or malformed codes.
    result = result[result["代码"].str.startswith(("60", "68", "00", "30"))]
    return result.drop_duplicates("代码").reset_index(drop=True)


def clean_history(frame):
    if frame is None or frame.empty:
        raise DataError("接口返回空行情")
    aliases = {"date": "Date", "日期": "Date", "open": "Open", "开盘": "Open",
               "high": "High", "最高": "High", "low": "Low", "最低": "Low",
               "close": "Close", "收盘": "Close", "volume": "Volume", "成交量": "Volume"}
    data = frame.rename(columns=aliases).copy()
    if "Date" in data:
        dates = pd.to_datetime(data.pop("Date"), errors="coerce")
        data.index = pd.DatetimeIndex(dates)
    elif not isinstance(data.index, pd.DatetimeIndex):
        raise DataError("缺少日期列")
    if data.index.tz is not None:
        data.index = data.index.tz_convert(CN).tz_localize(None)
    data.index = data.index.normalize()
    missing = set(PRICE_COLUMNS) - set(data.columns)
    if missing:
        raise DataError("缺少行情字段：" + ", ".join(sorted(missing)))
    data = data[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data[~data.index.isna()].dropna()
    data = data[~data.index.duplicated(keep="last")].sort_index()
    valid = ((data[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
             & (data["Volume"] >= 0)
             & (data["High"] >= data[["Open", "Close", "Low"]].max(axis=1))
             & (data["Low"] <= data[["Open", "Close", "High"]].min(axis=1)))
    data = data[valid & (data.index <= pd.Timestamp(now_cn().date()))]
    if data.empty:
        raise DataError("行情校验后无有效记录")
    data.index.name = "Date"
    return data


def error_text(error):
    if isinstance(error, requests.Timeout):
        return "请求超时"
    if isinstance(error, requests.HTTPError):
        return f"HTTP {error.response.status_code}（限流或访问受限）"
    if isinstance(error, requests.ConnectionError):
        return "网络连接失败（检查部署地区、DNS 或数据源可用性）"
    return f"{type(error).__name__}: {str(error)[:200]}"


class MarketClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json,*/*"})
        retry = Retry(total=1, connect=1, read=0, status=1, backoff_factor=0.3,
                      status_forcelist=[429, 502, 503, 504], allowed_methods=["GET"],
                      respect_retry_after_header=False, raise_on_status=False)
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def close(self):
        self.session.close()

    def get(self, url, **kwargs):
        response = self.session.get(url, timeout=(3.05, 8), **kwargs)
        response.raise_for_status()
        if len(response.content) > 5_000_000:
            raise DataError("数据源响应过大")
        return response

    def tencent(self, ticker, kind):
        code, market = ticker.split(".")
        symbol = ("sh" if market == "SS" else "sz") + code
        adjust = "" if kind == "index" else "qfq"
        response = self.get("https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get",
                            params={"param": f"{symbol},day,,,320,{adjust}"},
                            headers={"Referer": "https://gu.qq.com/"}).json()
        response = require_object(response, "腾讯")
        if response.get("code") != 0:
            raise DataError("腾讯返回异常状态")
        mapping = require_object(response.get("data") or {}, "腾讯行情")
        payload = require_object(mapping.get(symbol) or {}, "腾讯标的")
        rows = payload.get(adjust + "day")
        adjustment = "不复权" if kind == "index" else "前复权"
        if not rows:
            rows = payload.get("day")
            adjustment = "不复权"
        if not rows:
            raise DataError("腾讯没有该标的日线")
        if not isinstance(rows, list) or any(not isinstance(row, list) or len(row) < 6 for row in rows):
            raise DataError("腾讯日线字段格式异常")
        data = pd.DataFrame([row[:6] for row in rows],
                            columns=["Date", "Open", "Close", "High", "Low", "Volume"])
        quote_map = payload.get("qt")
        quote = quote_map.get(symbol) if isinstance(quote_map, dict) else None
        name = str(quote[1]) if isinstance(quote, list) and len(quote) > 1 else ticker
        return data, name, adjustment

    def eastmoney(self, ticker, kind):
        code, market = ticker.split(".")
        start = (pd.Timestamp(now_cn().date()) - pd.Timedelta(days=600)).strftime("%Y%m%d")
        payload = self.get("https://push2his.eastmoney.com/api/qt/stock/kline/get", params={
            "secid": f"{'1' if market == 'SS' else '0'}.{code}", "klt": "101",
            "fqt": "0" if kind == "index" else "1", "beg": start, "end": "20500101",
            "lmt": "320", "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        }, headers={"Referer": "https://quote.eastmoney.com/"}).json()
        payload = require_object(payload, "东方财富")
        detail = require_object(payload.get("data") or {}, "东方财富行情")
        rows = detail.get("klines") or []
        if not rows:
            raise DataError("东方财富没有该标的日线")
        if not isinstance(rows, list) or any(not isinstance(row, str) or len(row.split(",")) < 6 for row in rows):
            raise DataError("东方财富日线字段格式异常")
        data = pd.DataFrame([row.split(",")[:6] for row in rows],
                            columns=["Date", "Open", "Close", "High", "Low", "Volume"])
        return data.tail(320), str(detail.get("name") or ticker), "不复权" if kind == "index" else "前复权"

    def history(self, ticker, kind="stock", provider="自动切换"):
        if kind not in ("stock", "index"):
            raise DataError("未知资产类型")
        if kind == "index":
            if ticker not in INDICES:
                raise DataError("不支持的指数")
        elif normalize_a_share_code(ticker) != ticker:
            raise DataError("无效的沪深 A 股／基金代码")
        sources = PROVIDERS if provider == "自动切换" else (provider,)
        attempts, stale = [], []
        for source in sources:
            try:
                if source not in PROVIDERS:
                    raise DataError("未知行情源")
                raw, name, adjustment = (self.tencent if source == "腾讯" else self.eastmoney)(ticker, kind)
                frame = clean_history(raw)
                if len(frame) < 80:
                    raise DataError(f"仅有 {len(frame)} 条有效日线，至少需要 80 条")
                result = History(frame, source, name, now_cn().isoformat(timespec="seconds"),
                                 adjustment, list(attempts))
                age = (now_cn().date() - frame.index[-1].date()).days
                if age > 7:
                    attempts.append(f"{source}：最近数据为 {frame.index[-1].date()}，尝试更新来源")
                    stale.append(result)
                    continue
                if kind == "stock" and adjustment == "不复权":
                    result.notes.append("接口仅返回不复权价格，除权除息可能影响指标")
                return result
            except (DataError, requests.RequestException, ValueError, KeyError, TypeError) as error:
                message = f"{source}：{error_text(error)}"
                LOG.warning("%s %s", ticker, message)
                attempts.append(message)
        if stale:
            best = max(stale, key=lambda item: item.frame.index[-1])
            best.notes = attempts + ["数据超过 7 个自然日未更新；仅供查看历史，模型评分已停用"]
            return best
        raise DataError("；".join(attempts))

    def constituents(self):
        attempts = []
        for source in ("中证指数", "新浪"):
            try:
                if source == "中证指数":
                    response = self.get("https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/closeweight/000300closeweight.xls")
                    raw = pd.read_excel(BytesIO(response.content), dtype=str)
                    date_col = next((c for c in raw.columns if str(c).startswith("日期")), None)
                    as_of = str(raw[date_col].iloc[0]) if date_col else "接口未提供"
                else:
                    pages = []
                    for page in range(1, 4):
                        payload = self.get("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData",
                                           params={"node": "hs300", "page": page, "num": 100,
                                                   "sort": "symbol", "asc": 1}).json()
                        if not isinstance(payload, list):
                            raise DataError("成分股列表格式改变")
                        pages.append(pd.DataFrame(payload))
                    raw, as_of = pd.concat(pages, ignore_index=True), "接口未提供（按抓取时间记录）"
                frame = normalize_constituent_frame(raw)
                if len(frame) != 300:
                    raise DataError(f"仅取得 {len(frame)} 只唯一成分股，未通过完整性检查")
                return {"frame": frame, "source": source, "as_of": as_of,
                        "fetched_at": now_cn().isoformat(timespec="seconds"), "notes": attempts}
            except (DataError, requests.RequestException, ValueError, KeyError, TypeError,
                    ImportError, XLRDError, BadZipFile) as error:
                attempts.append(f"{source}：{error_text(error)}")
        raise DataError("；".join(attempts) + "。请稍后重试，或使用单股诊断。")


def load_history(ticker, kind="stock", provider="自动切换"):
    client = MarketClient()
    try:
        return client.history(ticker, kind, provider)
    finally:
        client.close()


def load_constituents():
    client = MarketClient()
    try:
        return client.constituents()
    finally:
        client.close()
