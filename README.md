# SENTINEL A-SHARE ADVANCED V27 · 全量排名版

基于用户提供的 `A_SHARE.txt` 完成重构，保留已修复的单股诊断，恢复红金深色界面、详细说明与四指数大盘追踪。沪深300扫描处理完整300只成分股，默认展示全部可评分股票的排名。

2026-09-29 配色更新：综合评分、涨跌与EV（启发式盈亏分值）采用正红负绿、零值灰蓝的连续色阶；概率与形态修正值采用低绿高红、50%为中点的固定色阶。浅色数字配实色背景，切换全部／前20／前50名时保持同一色标。该更新只涉及页面显示，行情、模型与依赖不变。

## 快速部署

逐步部署说明见 [DEPLOYMENT.md](DEPLOYMENT.md)。现有仓库推荐继续使用 `streamlit_app.py` 入口。

1. 将本目录内容放入 **A 股项目**的 GitHub 仓库，同级保留 `app.py`、`market_data.py`、`analysis_engine.py`、`app_services.py`、新增的 `ui_components.py` 与 `requirements.txt`，同时上传 `.streamlit/config.toml`。不要只复制 `app.py`。
2. Streamlit Community Cloud 指向该仓库和对应分支，入口设置为 `streamlit_app.py`，Python 选择 **3.12**。
3. 部署完成后先诊断 `600519 000001 510300`，核对最近价、数据源、行情日期和评分日期，再运行300股扫描。
4. 现有 `lzk00099/sentinel-a-share` 仓库的 `streamlit_app.py` 与附件内容完全一致。本包保留 `streamlit_app.py` 和 `sentinel_a_share_v26_fixed.py` 两个兼容入口，原部署使用它们时无需改入口设置。完整复制本包代码与模块即可。
5. 若云端依赖更新后仍未生效，在 Streamlit 的管理页面重启该 A 股应用。云端网络状况必须在部署后核验。

不需要 API Key、Token 或 Streamlit Secrets。已锁定本次实际测试使用的主要依赖版本。Python 3.12 为本次验证环境。

Streamlit 的文件组织与部署方式参考[官方部署文档](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)；更新后的重启操作参考[官方重启说明](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/reboot-your-app)。

## 本地运行

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

运行测试：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## 解决了什么

| 原问题 | 修复行为 |
|---|---|
| `normalize_constituent_frame(None)` 在空值检查前访问 `.columns` | 先检查空值与空表，成分名单读取失败可以进入备用源 |
| 个股和大盘仅通过 Yahoo 读取 | 腾讯日线为主，东方财富为备；可以手动指定来源 |
| `except: return None` 隐藏网络和模型错误 | 区分行情读取、指数读取、模型计算阶段，展示逐股错误并支持下载 |
| 线程等待超时后底层网络仍可能继续 | 请求自身设连接／读取超时，最多一次重试，不为每次请求新建超时线程 |
| 全市场股票名称字典阻塞启动 | 首屏只读取四个指数；股票名称随标的行情返回 |
| 部分成分名单被当作全量扫描 | 必须取得300只唯一成分股；不完整则换源，全部失败则明确停止 |
| 按钮运行结果随控件重运行消失 | 使用会话状态保存报告；报告显示创建时间、来源与日期 |
| 只扫描30只、扫描时限截断整池 | 完整处理300只，最多3个并发请求，无固定3分钟截断；仅连续6只读取失败时保存进度并暂停，可继续 |
| 界面过于简略、缺少大盘追踪 | 红金深色界面、详细策略说明、四指数卡片及走势；大盘约每60秒刷新 |
| 展示范围与扫描数量混淆 | 默认全部排名；前20／50名仅改变展示，CSV始终导出全部已评分结果 |
| 未完成的未来标签被转换为0 | 最近5条未知未来结果保持缺失，不进入训练 |
| 单一训练类别导致 `predict_proba()[1]` 越界 | 根据模型的实际类别提取正类概率，并提示参考性有限 |
| ATR 只算当日高低差 | 使用包含跳空的真实波幅 |
| 停牌零成交量被前向填充成正常成交量 | 不制造价格或成交量；异常行剔除，最近零成交量暂停评分 |
| 日线最后一行一律描述为实时高频数据 | 分别展示行情日期和评分日期；15:15前排除当日未收盘日线 |

## 数据与模型口径

- 行情为公开日线接口快照，可能延迟，不保证实时。每只标的使用一个来源的整段序列，不拼接不同来源的价格或成交量。
- 股票／场内基金优先使用前复权；若腾讯仅返回不复权数据会标注。指数不复权。复权历史价格与当时实际成交价可能不同。
- 成交量保留各来源原始单位，仅用于同一序列内的相对量比，不作为跨来源绝对成交量比较。行情明细中的 `Volume` 同样遵循此口径。
- 中证指数名单可能是定期发布的快照，页面显示文件标注日期。备用新浪接口未提供名单日期，页面明确说明。
- 支持沪深 A 股及常见场内基金。当前不支持北交所、B股、港股或美股。代码格式合法不代表资产一定存在，具体以接口返回为准。
- 模型使用最多约320条日线，至少85条完整记录、60条有效训练样本。训练目标为未来5条日线最高价是否超过当前收盘价加1.5倍ATR。
- “模型突破概率”未经过独立样本外校准。“形态修正值”“启发式盈亏分值”保留原策略的排序思路，不是已验证胜率或可兑现收益。参考价格区间不与训练标签的收益事件等价，因此不再称为“数学期望收益”。
- 四个指数缺失、过旧或日期不一致时，停用市场环境乘数调整。沪深300基准不可用或日期无法对齐时，仅使用个股特征，并在结果提示。
- 日线超过7个自然日未更新时仍允许查看取得的历史序列，但暂停模型评分；长假、停牌也可能触发此规则。此规则不是交易所日历判断。
- 个股行情成功缓存10分钟，指数成功缓存60秒，成分股成功缓存6小时；异常不会作为空数据长期缓存。清缓存不删除已生成的报告，报告仍保留原始时间。
- 单次网络请求的连接超时为3.05秒、读取超时为8秒；最多重试一次。读取超时是套接字等待限制，不是全局任务硬截止。扫描持续至整池检查完毕；持续读取失败时保留进度，恢复网络后可继续。
- 首页追踪沪深300、上证指数、创业板指、中证500，可展开20／60／120个交易日的归一化走势对比。刷新大盘不重跑个股模型，报告保留评分时的市场环境快照。
- 扫描覆盖全部300只，并不保证每只都能评分；过旧行情、停牌、样本不足等异常单独列出，可重试失败标的。排名按综合评分降序，同分按代码排序，未检查完的结果明确标注为暂定排名。

缓存与会话状态的实现遵循 [Streamlit 缓存文档](https://docs.streamlit.io/develop/concepts/architecture/caching)和[会话状态文档](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)。数据接口字段核对参考 [AKShare 腾讯行情实现](https://github.com/akfamily/akshare/blob/main/akshare/stock_feature/stock_hist_tx.py)、[指数行情实现](https://github.com/akfamily/akshare/blob/main/akshare/index/index_stock_zh.py)和[成分股实现](https://github.com/akfamily/akshare/blob/main/akshare/index/index_cons.py)；本项目直接发送有超时限制的公开 HTTP 请求，不依赖 AKShare／yfinance 的运行时调用。

## 文件说明

- `app.py`：页面、扫描进度、异常明细、结果导出。
- `streamlit_app.py`、`sentinel_a_share_v26_fixed.py`：每次重运行都会执行新版页面的兼容入口。
- `market_data.py`：输入规范化、HTTP请求、双源读取、行情与成分名单校验。
- `analysis_engine.py`：完整日线、特征、训练标签和评分。
- `app_services.py`：成功数据缓存。
- `ui_components.py`：红金主题、四指数卡片、详细使用说明与全量排行榜。
- `tests/`：数据读取、算法边界、页面交互的自动化测试。
- `VALIDATION.md`：实际完成的验证与尚未验证的范围。
- `DEPLOYMENT.md`：上传文件、Streamlit 配置与常见报错的处理步骤。

## 云端仍失败时

先下载页面异常记录，查看具体标的、阶段、来源与错误。请求超时、HTTP 403/429 与模型样本不足是不同问题。可切换来源或清缓存重试；持续的云端网络限制需要调整部署网络或接入具备访问授权的数据服务，不能仅靠页面代码保证所有网络都可访问。

此交付包是代码修复成果。是否已部署应以目标 GitHub 分支和 Streamlit 页面实际版本为准。
