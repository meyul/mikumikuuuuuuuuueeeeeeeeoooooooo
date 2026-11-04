"""yulstar: ZIP 원본에서 집계하는 GitHub 개발 활동 탐구 대시보드."""
from pathlib import Path
import io
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent
COLUMNS = ['repo', 'commit_hash', 'author_datetime_utc', 'commit_type',
           'author_type', 'ai_assisted', 'ai_tool', 'is_merge']


def truth(series):
    s = series.fillna('').astype(str).str.strip().str.lower()
    return s.map({'true': '표시 있음', '1': '표시 있음', 'yes': '표시 있음',
                  'false': '표시 없음', '0': '표시 없음', 'no': '표시 없음'}).fillna('알 수 없음')


def read_sources(sources):
    """필요한 열만 읽고, 저장소+해시로 전체 파일 사이의 중복도 제거한다."""
    frames, details = [], []
    for name, source in sources:
        members = []
        if name.lower().endswith('.zip'):
            with zipfile.ZipFile(source) as archive:
                for member in archive.infolist():
                    if member.filename.lower().endswith('.csv') and not member.is_dir():
                        with archive.open(member) as f:
                            members.append(pd.read_csv(f, usecols=lambda c: c in COLUMNS,
                                                       dtype='string', encoding='utf-8-sig'))
            if not members:
                raise ValueError(f'{name}: ZIP 안에 CSV가 없습니다.')
        else:
            members.append(pd.read_csv(source, usecols=lambda c: c in COLUMNS,
                                       dtype='string', encoding='utf-8-sig'))
        for frame in members:
            required = {'repo', 'commit_hash', 'author_datetime_utc'}
            if not required.issubset(frame.columns):
                raise ValueError(f'{name}: 필수 열이 없습니다: {sorted(required - set(frame.columns))}')
            details.append({'파일': name, '행 수': len(frame)})
            frames.append(frame)
    if not frames:
        raise ValueError('읽을 CSV 또는 ZIP이 없습니다.')
    data = pd.concat(frames, ignore_index=True)
    raw = len(data)
    for col in COLUMNS:
        if col not in data:
            data[col] = pd.NA
    invalid_id = data['repo'].fillna('').str.strip().eq('') | data['commit_hash'].fillna('').str.strip().eq('')
    missing_ids = int(invalid_id.sum())
    data = data.loc[~invalid_id].copy()
    dates = pd.to_datetime(data['author_datetime_utc'], errors='coerce', utc=True, format='mixed')
    invalid_dates = int(dates.isna().sum())
    data['month'] = dates.dt.tz_localize(None).dt.to_period('M').dt.to_timestamp()
    data = data.loc[data['month'].notna()].copy()
    duplicates = int(data.duplicated(['repo', 'commit_hash']).sum())
    data = data.drop_duplicates(['repo', 'commit_hash'])
    data['ai_assisted'] = truth(data['ai_assisted'])
    data['is_merge'] = truth(data['is_merge'])
    for col in ['commit_type', 'author_type', 'ai_tool']:
        data[col] = data[col].fillna('').str.strip().replace('', '미분류')
    data = data.drop(columns=['author_datetime_utc', 'commit_hash'])
    for col in ['repo', 'commit_type', 'author_type', 'ai_tool', 'ai_assisted', 'is_merge']:
        data[col] = data[col].astype('category')
    audit = {'원본 행 수': raw, '식별값 누락 제외': missing_ids, '날짜 오류 제외': invalid_dates,
             '중복 제외': duplicates, '분석 가능 행 수': len(data)}
    return data, audit, pd.DataFrame(details)


@st.cache_data(show_spinner=False, max_entries=2)
def load_local(signature):
    return read_sources([(p, ROOT / 'data' / p) for p, _, _ in signature])


@st.cache_data(show_spinner=False, max_entries=1)
def load_upload(payloads):
    return read_sources([(name, io.BytesIO(payload)) for name, payload in payloads])


def compare(data, boundary, months, zero_fill):
    start = boundary - pd.DateOffset(months=months)
    end = boundary + pd.DateOffset(months=months)
    calendar = pd.date_range(start, periods=2 * months, freq='MS')
    # 비교창의 첫 달 이전 및 마지막 달 이후까지 관측되는 동일 저장소만 선택.
    spans = data.groupby('repo', observed=True)['month'].agg(['min', 'max'])
    eligible = spans.index[(spans['min'] <= start) & (spans['max'] >= calendar[-1])]
    scoped = data[data['repo'].isin(eligible) & data['month'].ge(start) & data['month'].lt(end)].copy()
    counts = scoped.groupby(['month', 'repo'], observed=True).size().unstack('repo')
    counts = counts.reindex(index=calendar, columns=eligible)
    empty_cells = int(counts.isna().sum().sum())
    if zero_fill:
        counts = counts.fillna(0)
    else:
        # 관측 누락을 0으로 간주하지 않을 때는 창 내 모든 달에 관측이 있는 저장소만 사용.
        counts = counts.loc[:, counts.notna().all()]
        scoped = scoped[scoped['repo'].isin(counts.columns)].copy()
    scoped['period'] = np.where(scoped['month'] < boundary, '이전', '이후')
    monthly = counts.sum(axis=1).rename('커밋 수').rename_axis('월').to_frame()
    monthly['구간'] = np.where(monthly.index < boundary, '이전', '이후')
    before = counts.iloc[:months].mean()
    after = counts.iloc[months:].mean()
    repositories = pd.DataFrame({'이전 월평균': before, '이후 월평균': after})
    repositories['차이'] = after - before
    repositories['변화율 (%)'] = np.where(before.ne(0), (after - before) / before * 100, np.nan)
    return scoped, monthly, repositories.rename_axis('저장소'), empty_cells


def model_comparison(monthly, boundary):
    y = monthly['커밋 수'].to_numpy(dtype=float)
    t = np.arange(len(y), dtype=float)
    post = (monthly.index >= boundary).astype(float)
    since = np.maximum(0, t - np.searchsorted(monthly.index, boundary))
    features = {'시간 선형회귀': t[:, None],
                '전후 구간 회귀': np.column_stack([t, post, since])}
    split = int(len(y) * .8)
    if split < 5 or len(y) - split < 2:
        raise ValueError('모델 평가를 위해 더 긴 기간을 선택하세요.')
    results, predictions = [], {}
    baseline = np.repeat(y[:split].mean(), len(y) - split)
    predictions['학습 평균 기준선'] = baseline
    for name, x in features.items():
        # 학습 구간에 기준 시점 전후 자료가 모두 있어야 구간 회귀를 평가할 수 있다.
        if name == '전후 구간 회귀' and len(np.unique(post[:split])) < 2:
            continue
        model = LinearRegression().fit(x[:split], y[:split])
        predictions[name] = np.maximum(0, model.predict(x[split:]))
    for name, pred in predictions.items():
        results.append({'모델': name, '평가 MAE': mean_absolute_error(y[split:], pred)})
    scores = pd.DataFrame(results).sort_values('평가 MAE').reset_index(drop=True)
    chart = pd.DataFrame(predictions, index=monthly.index[split:])
    chart.insert(0, '실제 커밋 수', y[split:])
    return scores, chart, split


def composition(data, column):
    result = pd.crosstab(data[column], data['period'], dropna=False).reindex(columns=['이전', '이후'], fill_value=0)
    for period in ['이전', '이후']:
        total = result[period].sum()
        result[f'{period} 비율 (%)'] = result[period] / total * 100 if total else 0.0
    result['비율 차이 (%p)'] = result['이후 비율 (%)'] - result['이전 비율 (%)']
    return result.sort_values('이후', ascending=False)


def csv_bytes(frame):
    return frame.to_csv(index=True).encode('utf-8-sig')


def main():
    st.set_page_config(page_title='yulstar · AI와 개발 활동', page_icon='✦', layout='wide')
    st.markdown('''<style>
    .stApp {background: #f7f9fe;} .block-container {max-width: 1250px; padding-top: 2rem;}
    [data-testid="stMetric"] {background:white; border:1px solid #e2e8f4; padding:18px; border-radius:14px;}
    h1,h2,h3 {letter-spacing:-.035em;} [data-testid="stSidebar"] {background:#eef2fc;}
    </style>''', unsafe_allow_html=True)
    st.caption('YULSTAR / GITHUB COMMIT EXPLORER')
    st.title('AI가 등장한 뒤, 개발 활동은 달라졌을까?')
    st.write('같은 저장소의 커밋 기록으로 활동량과 작업 구성을 비교하는 탐구 대시보드')
    st.info('커밋 수는 개발 활동량의 지표입니다. 작업 종류는 코드 품질을 직접 측정하지 않으며, 전후 차이만으로 AI의 효과를 증명할 수 없습니다.')
    with st.sidebar:
        st.header('분석 설정')
        mode = st.radio('데이터', ['저장소 데이터', 'CSV / ZIP 직접 업로드'])
        uploads = st.file_uploader('여러 연도 파일을 함께 선택', type=['csv', 'zip'], accept_multiple_files=True) if mode.endswith('업로드') else []
    try:
        with st.spinner('ZIP 안의 CSV를 읽고 중복·날짜를 확인하는 중…'):
            if mode == '저장소 데이터':
                files = sorted(p for p in (ROOT / 'data').glob('*') if p.suffix.lower() in {'.zip', '.csv'})
                signature = tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files)
                data, audit, sources = load_local(signature)
            else:
                if not uploads:
                    st.info('왼쪽에서 CSV 또는 ZIP을 선택하세요.'); return
                data, audit, sources = load_upload(tuple((p.name, p.getvalue()) for p in uploads))
    except (ValueError, OSError, zipfile.BadZipFile, pd.errors.ParserError, UnicodeError) as exc:
        st.error(f'데이터를 읽지 못했습니다: {exc}'); return
    if data.empty:
        st.warning('전처리 후 분석할 커밋이 없습니다.'); return
    with st.sidebar:
        boundary_date = st.date_input('AI 대중화 기준 월', value=pd.Timestamp('2022-12-01').date(),
                                      help='선택한 날짜가 속한 달부터 이후로 분류합니다. 기본값은 2022년 12월입니다.')
        boundary = pd.Timestamp(boundary_date).to_period('M').to_timestamp()
        months = st.slider('전후 각각 비교할 개월 수', 6, 36, 24, step=6)
        options = sorted(data['repo'].astype(str).unique())
        selected = st.multiselect('저장소 선택 (비우면 전체)', options)
        exclude_merge = st.checkbox('병합 커밋 제외', value=False)
        zero_fill = st.checkbox('관측 사이의 빈 달을 커밋 0건으로 가정', value=False,
                                help='기본값은 빈 달이 있는 저장소를 제외합니다. 원본이 완전한 Git 히스토리라고 확인한 경우에만 켜세요.')
    filtered = data[data['repo'].isin(selected)].copy() if selected else data.copy()
    if exclude_merge:
        filtered = filtered[filtered['is_merge'] != '표시 있음']
    scoped, monthly, repos, empty_cells = compare(filtered, boundary, months, zero_fill)
    with st.expander('데이터와 전처리 내역', expanded=False):
        st.write(audit)
        st.dataframe(sources, hide_index=True, width="stretch")
        st.write(f"관측 범위: {data['month'].min():%Y-%m} ~ {data['month'].max():%Y-%m} · 원본 저장소 {data['repo'].nunique():,}개")
        st.caption('작성자 ID·커밋 제목 등 개인 식별에 사용될 수 있는 열은 읽지 않습니다. 중복 키는 저장소+커밋 해시입니다. 작성 시각(UTC)을 사용합니다. 마지막 관측 월이 완전히 수집됐는지는 원본 설명으로 확인해야 합니다.')
        st.caption('AI 관련 미표시는 AI 미사용의 증거가 아닙니다. 작업 종류 누락은 미분류로 남겨 비율의 분모에 포함합니다.')
    st.caption(f'비교: {monthly.index.min():%Y-%m} ~ {(boundary - pd.DateOffset(months=1)):%Y-%m} / {boundary:%Y-%m} ~ {monthly.index.max():%Y-%m}. 각 {months}개월, 같은 저장소를 비교합니다.')
    st.caption('비교창 첫 달과 마지막 달까지 기록이 이어지는 저장소를 먼저 선택합니다. 빈 달 제외 모드는 매달 커밋이 있는 활발한 저장소에 치우칠 수 있습니다.')
    if repos.empty:
        st.warning('조건을 충족하는 저장소가 없습니다. 기간을 줄이거나 저장소 선택을 바꾸세요. 완전한 히스토리임을 확인했다면 빈 달 0건 가정을 사용할 수 있습니다.'); return
    if zero_fill:
        st.warning(f'원본 히스토리가 완전하다는 가정으로 저장소·월 조합 {empty_cells:,}개의 빈 값을 0건으로 처리했습니다. 누락 여부는 데이터만으로 확인할 수 없습니다.')
    before = monthly.iloc[:months]['커밋 수'].mean()
    after = monthly.iloc[months:]['커밋 수'].mean()
    change = (after / before - 1) * 100 if before else None
    cards = st.columns(4)
    cards[0].metric('비교 저장소', f'{len(repos):,}개')
    cards[1].metric('이전 월평균', f'{before:,.1f}건')
    cards[2].metric('이후 월평균', f'{after:,.1f}건')
    cards[3].metric('월평균 변화율', f'{change:+.1f}%' if change is not None else '계산 불가')
    tabs = st.tabs(['커밋 활동량', '작업 구성 · AI', '회귀 모델', '탐구 결과 내려받기'])
    with tabs[0]:
        fig = px.line(monthly.reset_index(), x='월', y='커밋 수', markers=True,
                      color_discrete_sequence=['#5262db'])
        fig.add_vline(x=boundary.timestamp()*1000, line_dash='dash', line_color='#ec7790')
        fig.update_layout(xaxis_title='', yaxis_title='월별 커밋 수', hovermode='x unified')
        st.plotly_chart(fig, width="stretch")
        st.subheader('저장소마다 변화가 같을까?')
        st.caption(f"증가 {(repos['차이'] > 0).sum()}개 · 감소 {(repos['차이'] < 0).sum()}개 · 동일 {(repos['차이'] == 0).sum()}개")
        st.dataframe(repos.sort_values('차이', ascending=False), width="stretch")
    with tabs[1]:
        column = st.selectbox('비교할 항목', ['commit_type', 'ai_assisted', 'author_type', 'ai_tool'],
                              format_func=lambda x: {'commit_type':'작업 종류','ai_assisted':'AI 도움 표시','author_type':'작성자 유형','ai_tool':'AI 도구 표시'}[x])
        table = composition(scoped, column)
        st.caption('AI 도구·작성자 유형은 원본 라벨을 그대로 사용합니다. 작업 종류는 품질 점수가 아닙니다.')
        top = table.head(20).reset_index().melt(id_vars=column, value_vars=['이전 비율 (%)', '이후 비율 (%)'], var_name='구간', value_name='비율 (%)')
        st.plotly_chart(px.bar(top, x=column, y='비율 (%)', color='구간', barmode='group',
                               color_discrete_sequence=['#b5c2ee', '#5262db']), width="stretch")
        st.caption('그래프는 이후 건수 상위 20개 항목, 표는 전체 항목입니다. 비율의 분모는 각 구간의 전체 커밋 수입니다.')
        st.dataframe(table, width="stretch")
    scores, predictions, split = model_comparison(monthly, boundary)
    with tabs[2]:
        st.write(f'시간순 앞 {split}개월 학습 / 뒤 {len(monthly)-split}개월 평가. MAE는 월별 예측이 실제 값에서 평균 몇 건 벗어났는지를 뜻합니다.')
        st.dataframe(scores, hide_index=True, width="stretch")
        st.plotly_chart(px.line(predictions, labels={'value':'커밋 수','index':'평가 월','variable':'모델'}), width="stretch")
        st.caption('시간 회귀: 월 순서. 전후 구간 회귀: 월 순서 + 기준 월 이후 여부 + 이후 경과 개월. 음수 예측은 0건으로 제한합니다. 평가 MAE가 가장 낮은 모델을 선택하지만, 별도 최종 시험 구간이 없으므로 선택 결과의 일반화 성능을 확정할 수 없습니다.')
    with tabs[3]:
        best = scores.iloc[0]
        baseline = scores.loc[scores['모델'].eq('학습 평균 기준선'), '평가 MAE'].iloc[0]
        delta_text = f'{change:+.2f}%' if change is not None else '계산 불가 (이전 0건)'
        report = f'''# yulstar 분석 결과

- 기준 월: {boundary:%Y-%m} / 전후 각각 {months}개월
- 비교 저장소: {len(repos)}개 / 비교 커밋: {len(scoped):,}건
- 이전 월평균: {before:,.2f}건 / 이후 월평균: {after:,.2f}건
- 월평균 변화율: {delta_text}
- 증가 저장소: {(repos['차이'] > 0).sum()}개 / 감소: {(repos['차이'] < 0).sum()}개
- 선택 모델: {best['모델']} / 평가 MAE: {best['평가 MAE']:,.2f}건
- 기준선 MAE: {baseline:,.2f}건
- 빈 달 처리: {'완전한 히스토리를 가정해 0건 처리' if zero_fill else '빈 달이 있는 저장소 제외'}
- 병합 커밋 제외: {exclude_merge}
- 저장소 선택: {', '.join(selected) if selected else '전체에서 조건 충족 저장소'}

## 전처리
{audit}

## 작업 종류별 전후 비교
{composition(scoped, 'commit_type').to_csv()}

## 해석의 범위
커밋 수는 개발 활동량을 나타내며 코드 품질이나 생산성을 직접 증명하지 않는다.
작업 종류 비율도 품질 점수가 아니다. AI 관련 라벨이 없어도 AI를 사용했을 수 있다.
시간 변화, 프로젝트 성장, 저장소 선정 편향을 통제하지 않았으므로 AI의 인과적 효과로 단정할 수 없다.
월별 관측 누락과 0건을 원본 데이터만으로 구분할 수 없다. 창 전체에서 활동한 저장소를 선택하는 데 따른 생존 편향이 있다.
평가 구간으로 모델을 선택했으며 별도 최종 시험 구간은 없다.
'''
        st.markdown(report.split('## 전처리')[0])
        st.download_button('결과 보고서 (.md)', report.encode('utf-8'), 'yulstar_report.md', 'text/markdown')
        tables = {'monthly.csv':monthly, 'repositories.csv':repos, 'commit_types.csv':composition(scoped,'commit_type'),
                  'ai_assisted.csv':composition(scoped,'ai_assisted'), 'models.csv':scores, 'predictions.csv':predictions,
                  'source_files.csv':sources, 'preprocessing.csv':pd.DataFrame([audit])}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for filename, frame in tables.items():
                archive.writestr(filename, csv_bytes(frame))
            archive.writestr('report.md', report)
        st.download_button('집계표 전체 (.zip)', buffer.getvalue(), 'yulstar_results.zip', 'application/zip')
        st.caption('CSV는 엑셀에서 한글이 깨지지 않도록 UTF-8 BOM으로 저장합니다. 원본 개인 식별 정보는 다운로드에 포함되지 않습니다.')


if __name__ == '__main__':
    main()
