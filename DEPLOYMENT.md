# 部署到 GitHub 与 Streamlit

本说明适用于 A 股仓库：<https://github.com/lzk00099/sentinel-a-share>。

## 1. 解压并上传文件

解压 `SENTINEL_A_SHARE_V27_colorfix.zip`，进入里面的 `sentinel-a-share` 文件夹。上传**该文件夹里面的文件**到仓库根目录，替换同名文件；不要只上传 ZIP，也不要只替换入口文件。

如果已经完整部署上一版V27，本次配色更新只需同时覆盖 `app.py` 和 `ui_components.py` 两个文件。入口、依赖和主题配置不需要调整。更新后重新生成一次诊断或扫描，即可看到综合评分、涨跌、概率、形态修正值和EV列的红绿渐变。

运行所需的文件结构如下：

```text
sentinel-a-share（GitHub 仓库根目录）
├── streamlit_app.py
├── app.py
├── market_data.py
├── analysis_engine.py
├── app_services.py
├── ui_components.py
├── requirements.txt
├── sentinel_a_share_v26_fixed.py
└── .streamlit/
    └── config.toml
```

`ui_components.py` 是本版新增的界面模块，必须上传。`.streamlit/config.toml` 是红金深色主题配置，不包含密钥；若上传工具未显示这个目录，请在 GitHub 中按此路径创建并复制配置内容，以保留完整配色。其余 README、测试和验证文件建议一并保存。

务必替换旧 `requirements.txt`，不要将新内容简单追加在旧依赖后面。

## 2. 配置 Streamlit

打开 <https://share.streamlit.io/>。

已有 A 股应用、且已经使用 Python 3.12 时：确认它指向 `lzk00099/sentinel-a-share` 和本次更新的分支。旧入口是 `streamlit_app.py` 或 `sentinel_a_share_v26_fixed.py` 的，都可以保留，两个兼容入口均包含在包中。代码提交后等待自动更新；必要时从管理页面重启应用。

新建应用时：点击 **Create app**，选择已经有 GitHub 应用，填写：

| 设置 | 填写内容 |
|---|---|
| Repository | `lzk00099/sentinel-a-share` |
| Branch | `main`（若上传到其他分支，请填写实际分支） |
| Main file path | `streamlit_app.py` |
| Advanced settings → Python version | `3.12` |
| Secrets | 留空，本版不需要 API Key |

保存设置后点击 **Deploy**，等待依赖安装和应用启动。[Streamlit 官方部署说明](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

**Python 版本要在创建部署时选择。** 已有应用不能通过普通重启更改 Python 版本；若现有版本不适合，应按 Streamlit 官方流程记录原配置并重新部署。为便于保留旧页面，也可以先建立一个新的测试部署，验证后再切换。[官方 Python 版本说明](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python)

## 3. 验证读取功能

页面标题应为 **SENTINEL A-SHARE ADVANCED V27**，采用红金深色配色，顶部显示沪深300、上证指数、创业板指、中证500四张大盘卡片，并能看到“单股诊断”“沪深300扫描”“数据说明”三个页签。

1. 行情源选择“自动切换”。
2. 在单股诊断中输入 `600519 000001 510300`，点击“开始诊断”。
3. 核对是否出现最近价、数据源、行情日期、评分日期和综合评分。
4. 非交易时间看到最近交易日的数据属于正常现象，不要求行情日期等于当天。
5. 在“沪深300扫描”中点击“开始沪深300扫描”，会自动处理全部300只，无需选择数量。完成后应显示“已检查300”，默认展示全部成功评分股票的排名。前20／50名仅改变显示数量，“下载全部排名 CSV”始终包含全部已评分结果。
6. 个别股票行情过旧或样本不足时，会列入异常记录，不强行评分。出现“继续扫描”时可处理剩余标的；整池结束后可点击“重试失败标的”，已有成功结果会保留。
7. 大盘约每60秒刷新一次，也可点击“刷新大盘”。它显示日线接口快照，可能延迟；刷新大盘不会重跑已完成的单股诊断或全量扫描。

## 4. 如果部署后报错

| 现象 | 处理方法 |
|---|---|
| `No module named market_data` 等 | 检查所有 Python 模块是否与入口文件在同一目录 |
| 提示找不到 `app.py` | 兼容入口会执行 `app.py`，两者必须一起上传 |
| 安装依赖失败 | 确认完整替换了 `requirements.txt`，查看部署日志中的第一条安装错误；本包按 Python 3.12 验证 |
| HTTP 429、403 或请求超时 | 在应用中查看具体来源与错误，切换另一行情源；云端网络限制不能靠重启保证解决 |
| 没有评分但有行情 | 查看“模型计算”阶段的错误，例如样本不足、停牌或数据过旧 |
| 仍显示旧页面 | 确认更新的是该应用绑定的仓库、分支和入口；等待依赖安装结束，必要时重启 |

现有验证来自本地环境。Streamlit Cloud 的网络可用性需在真实部署后检查。
