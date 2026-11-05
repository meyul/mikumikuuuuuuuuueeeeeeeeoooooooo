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
           'author_type', 'ai_assisted', 'ai_tool', 'ai_model', 'is_merge',
           'category', 'n_coauthors']


def truth(series):
    s = series.fillna('').astype(str).str.strip().str.lower()
    return s.map({'true': '표시 있음', '1': '표시 있음', 'yes': '표시 있음',
                  'false': '표시 없음', '0': '표시 없음', 'no': '표시 없음'}).fillna('알 수 없음')


def data_profile(data):
    """결측 표시와 공백만 있는 셀을 함께 세되, 0과 FALSE는 빈 값으로 세지 않는다."""
    return pd.DataFrame([{'열 이름': col, '자료형': str(data[col].dtype),
                          '빈 값 개수': int(data[col].isna().sum() +
                              data[col].astype('string').str.strip().eq('').fillna(False).sum())}
                         for col in data.columns])


def read_sources(sources):
    """원본의 모든 열을 읽고, 저장소+해시로 파일 사이의 중복을 제거한다."""
    frames, details = [], []
    for name, source in sources:
        members = []
        if name.lower().endswith('.zip'):
            with zipfile.ZipFile(source) as archive:
                for member in archive.infolist():
                    if member.filename.lower().endswith('.csv') and not member.is_dir():
                        with archive.open(member) as f:
                            members.append(pd.read_csv(f, dtype='string', encoding='utf-8-sig'))
            if not members:
                raise ValueError(f'{name}: ZIP 안에 CSV가 없습니다.')
        else:
            members.append(pd.read_csv(source, dtype='string', encoding='utf-8-sig'))
        for frame in members:
            required = {'repo', 'commit_hash', 'author_datetime_utc'}
            if not required.issubset(frame.columns):
                raise ValueError(f'{name}: 필수 열이 없습니다: {sorted(required - set(frame.columns))}')
            details.append({'파일': name, '행 수': len(frame), '열 수': len(frame.columns)})
            frames.append(frame)
    if not frames:
        raise ValueError('읽을 CSV 또는 ZIP이 없습니다.')
    data = pd.concat(frames, ignore_index=True)
    raw = len(data)
    raw_columns = len(data.columns)
    raw_profile = data_profile(data)
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
    for col in ['commit_type', 'author_type', 'ai_tool', 'ai_model', 'category', 'n_coauthors']:
        data[col] = data[col].fillna('').str.strip().replace('', '미분류')
    for col in ['repo', 'commit_type', 'author_type', 'ai_tool', 'ai_model',
                'category', 'n_coauthors', 'ai_assisted', 'is_merge']:
        data[col] = data[col].astype('category')
    audit = {'원본 행 수': raw, '식별값 누락 제외': missing_ids, '날짜 오류 제외': invalid_dates,
             '중복 제외': duplicates, '분석 가능 행 수': len(data), '원본 열 수': raw_columns,
             '처리 후 열 수': len(data.columns), '원본 열 개요': raw_profile.to_dict('records')}
    return data, audit, pd.DataFrame(details)


@st.cache_data(show_spinner=False, max_entries=2)
def load_local(signature):
    return read_sources([(p, ROOT / 'data' / p) for p, _, _ in signature])


@st.cache_data(show_spinner=False, max_entries=1)
def load_upload(payloads):
    return read_sources([(name, io.BytesIO(payload)) for name, payload in payloads])


def detect_monthly_outliers(data, multiplier=5.0):
    """비교 대상의 관측 월만 사용해 저장소별 급증 후보를 한 번 탐지한다."""
    columns = ['저장소', '월', '커밋 수', '월 중앙값', '상한 기준', '중앙값 대비 배수']
    if data.empty:
        return pd.DataFrame(columns=columns)
    counts = data.groupby(['repo', 'month'], observed=True).size().rename('커밋 수').reset_index()
    groups = counts.groupby('repo', observed=True)['커밋 수']
    median = groups.transform('median')
    q1 = groups.transform(lambda s: s.quantile(.25))
    q3 = groups.transform(lambda s: s.quantile(.75))
    limit = pd.concat([q3 + 3 * (q3 - q1), median * multiplier,
                       pd.Series(100, index=counts.index)], axis=1).max(axis=1)
    # 짧거나 드문 관측으로 정상 활동을 이상치로 분류하지 않는다.
    flagged = groups.transform('size').ge(12) & counts['커밋 수'].gt(limit)
    result = counts.loc[flagged].rename(columns={'repo': '저장소', 'month': '월'}).copy()
    result['월 중앙값'] = median.loc[flagged]
    result['상한 기준'] = limit.loc[flagged]
    result['중앙값 대비 배수'] = result['커밋 수'] / result['월 중앙값']
    return result[columns].sort_values('커밋 수', ascending=False).reset_index(drop=True)


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


def cleaning_steps(audit):
    before = audit['원본 행 수']
    rows = []
    for label, key, method in [
        ('필수 식별값 결측', '식별값 누락 제외', 'repo 또는 commit_hash가 빈 행 제외'),
        ('작성일 결측·오류', '날짜 오류 제외', '작성일이 비었거나 날짜로 해석할 수 없는 행 제외'),
        ('중복', '중복 제외', '저장소 + 커밋 해시가 같은 행 중 첫 행 유지')]:
        removed = audit[key]
        rows.append({'단계': label, '처리 방법': method, '처리 전 행 수': before,
                     '제외 행 수': removed, '처리 후 행 수': before - removed})
        before -= removed
    rows.append({'단계': '선택 항목 결측', '처리 방법': '분류 항목은 미분류, AI 도움·병합 여부는 알 수 없음으로 표시. 그 밖의 원본 결측은 유지',
                 '처리 전 행 수': before, '제외 행 수': 0, '처리 후 행 수': before})
    return pd.DataFrame(rows)


def render_data_quality(data, audit):
    st.subheader('데이터 개요 · 결측치 처리')
    cards = st.columns(4)
    cards[0].metric('원본 행 수', f"{audit['원본 행 수']:,}")
    cards[1].metric('원본 열 수', f"{audit['원본 열 수']:,}")
    cards[2].metric('처리 후 행 수', f'{len(data):,}')
    cards[3].metric('처리 후 열 수', f'{len(data.columns):,}')
    original = pd.DataFrame(audit['원본 열 개요']).rename(
        columns={'자료형': '처리 전 자료형', '빈 값 개수': '처리 전 빈 값 개수'})
    current = data_profile(data).rename(columns={'자료형': '처리 후 자료형', '빈 값 개수': '처리 후 빈 값 개수'})
    profile = original.merge(current, on='열 이름', how='outer')
    profile['열 구분'] = np.where(profile['처리 전 자료형'].isna(), '분석용 추가 열', '원본 열')
    st.caption('빈 값 = NA·빈 문자열·공백만 있는 셀. 0과 FALSE는 유효한 값입니다. '
               '읽을 때 원문 보존을 위해 string 자료형을 사용합니다. 처리 후 category는 범주형, month는 날짜형입니다. '
               '분석용 추가 열은 원본에 없으므로 처리 전 칸이 비어 있습니다.')
    st.dataframe(profile, hide_index=True, width='stretch')
    st.dataframe(cleaning_steps(audit), hide_index=True, width='stretch')
    st.caption('선택 항목은 commit_type·author_type·ai_tool·ai_model·category·n_coauthors입니다. '
               'AI 도움·병합 여부에서 빈 값 또는 인식하지 못한 값은 알 수 없음으로 표시합니다. '
               '미분류·알 수 없음은 정보를 새로 알아낸 것이 아니라 결측·미확인 상태를 표시한 것입니다.')
    st.download_button('열별 자료형·빈 값 개수 (.csv)', csv_bytes(profile), 'column_overview.csv', 'text/csv')
    return profile


def render_activity(monthly, boundary, comparable=True):
    title = 'AI 대중화 전후 월별 커밋 활동량' if comparable else '관측 기간의 월별 커밋 활동량'
    fig = px.line(monthly.reset_index(), x='월', y='커밋 수', markers=True, title=title,
                  labels={'월': '커밋 작성 월 (UTC)', '커밋 수': '월별 커밋 수 (건)'},
                  color_discrete_sequence=['#5262db'])
    fig.add_vline(x=boundary.timestamp()*1000, line_dash='dash', line_color='#ec7790')
    fig.update_layout(xaxis_title='커밋 작성 월 (UTC)', yaxis_title='월별 커밋 수 (건)', hovermode='x unified')
    st.plotly_chart(fig, width='stretch')
    if comparable:
        before = monthly.loc[monthly.index < boundary, '커밋 수'].mean()
        after = monthly.loc[monthly.index >= boundary, '커밋 수'].mean()
        delta = after - before
        direction = '증가' if delta > 0 else '감소' if delta < 0 else '동일'
        change = f' ({delta / before * 100:+.1f}%)' if before else ''
        st.write(f'발견: 같은 저장소의 월평균 커밋 수는 기준 월 이후 {direction}. '
                 f'근거: 이전 {before:,.1f}건 → 이후 {after:,.1f}건{change}.')
    else:
        peak = monthly['커밋 수'].idxmax()
        st.write(f"발견: 관측 기간 중 {peak:%Y-%m}의 커밋 수가 가장 많음. "
                 f"근거: 해당 월 {monthly.loc[peak, '커밋 수']:,.0f}건, 관측 월 {len(monthly):,}개 합계 {monthly['커밋 수'].sum():,.0f}건.")
        st.caption('동일 저장소 전후 비교 조건이 충족되지 않아 관측 건수만 표시합니다. 기록이 없는 달은 0건으로 만들지 않습니다.')


def composition_range(data, column, start, end, boundary, hide_unknown=False):
    """종료 월을 포함하며, 전후 비교의 동일 저장소 선정과 별도로 탐색한다."""
    if start > end:
        raise ValueError('시작 월은 종료 월보다 늦을 수 없습니다.')
    selected = data.loc[data['month'].between(start, end)].copy()
    total = len(selected)
    unknown = int(selected[column].isin(['미분류', '알 수 없음']).sum())
    if hide_unknown:
        selected = selected.loc[~selected[column].isin(['미분류', '알 수 없음'])].copy()
    selected['period'] = np.where(selected['month'] < boundary, '이전', '이후')
    return composition(selected, column), total, unknown


def render_composition(data, boundary):
    st.subheader('막대그래프 분석 범위')
    st.caption('시작·종료 월을 선택하면 그 기간의 데이터를 다시 집계합니다. 종료 월도 포함합니다. '
               '기본값은 전체 관측 기간입니다. 저장소 선택·병합 제외·이상치 저장소 제외는 적용하고, '
               '전후 내내 활동한 동일 저장소 조건은 적용하지 않습니다.')
    calendar = pd.date_range(data['month'].min(), data['month'].max(), freq='MS').tolist()
    controls = st.columns(2)
    start = controls[0].selectbox('막대그래프 시작 월', calendar, index=0,
                                 format_func=lambda x: x.strftime('%Y-%m'), key='bar_start')
    end_options = [m for m in calendar if m >= start]
    end = controls[1].selectbox('막대그래프 종료 월', end_options, index=len(end_options)-1,
                               format_func=lambda x: x.strftime('%Y-%m'), key='bar_end')
    labels = {'ai_model': 'AI 모델명', 'ai_tool': 'AI 도구', 'ai_assisted': 'AI 도움 표시',
              'commit_type': '작업 종류', 'author_type': '작성자 유형', 'category': '저장소 분야',
              'n_coauthors': '공동 작성자 수', 'is_merge': '병합 여부', 'repo': '저장소'}
    column = st.selectbox('비교할 항목', list(labels), format_func=labels.get, key='bar_column')
    hide_unknown = st.checkbox('미분류·알 수 없음 제외하고 보기', value=False, key='bar_hide_unknown')
    display = st.radio('막대그래프 표시', ['커밋 건수', '구간 내 비율 (%)'], horizontal=True, key='bar_measure')
    table, total, unknown = composition_range(data, column, start, end, boundary, hide_unknown)
    st.caption(f'선택 범위 {start:%Y-%m} ~ {end:%Y-%m} · 전체 {total:,}건 · '
               f'선택 항목의 미분류·알 수 없음 {unknown:,}건 · 기준 월 {boundary:%Y-%m}부터 이후 구간')
    st.caption('원본에 기재된 도구명·모델명을 그대로 사용합니다. 모델명 누락은 AI 미사용을 뜻하지 않습니다.')
    if total == 0 or table[['이전', '이후']].to_numpy().sum() == 0:
        st.info('선택한 기간·항목에 표시할 기록이 없습니다. 기간을 바꾸거나 미분류 제외를 해제하세요.')
        return
    table = table.loc[(table['이전'] + table['이후']).gt(0)]
    table = table.assign(_total=table['이전'] + table['이후']).sort_values('_total', ascending=False).drop(columns='_total')
    count = st.slider('표시할 항목 수', 1, max(2, len(table)), min(20, max(2, len(table))), key='bar_top_n')
    values = ['이전', '이후'] if display == '커밋 건수' else ['이전 비율 (%)', '이후 비율 (%)']
    top = table.head(count).reset_index().melt(id_vars=column, value_vars=values,
                                              var_name='구간', value_name=display)
    st.plotly_chart(px.bar(top, x=column, y=display, color='구간', barmode='group',
                           title=f'{labels[column]}별 커밋 구성 ({start:%Y-%m} ~ {end:%Y-%m})',
                           labels={column: labels[column], display: '커밋 수 (건)' if display == '커밋 건수' else '구간 내 커밋 비율 (%)'},
                           color_discrete_sequence=['#b5c2ee', '#5262db']), width='stretch')
    sums = table['이전'] + table['이후']
    leader, leader_count = sums.idxmax(), int(sums.max())
    denominator = int(sums.sum())
    st.write(f'발견: 선택 기간에서 {labels[column]}의 최다 항목은 {leader}. '
             f'근거: {leader_count:,}건 / 표시 대상 {denominator:,}건 ({leader_count / denominator * 100:.1f}%).')
    st.caption('막대그래프를 드래그하면 화면만 확대됩니다. 집계 기간 변경은 위 시작·종료 월에서 합니다. '
               f'비율의 분모는 각 구간의 {"분류된" if hide_unknown else "전체"} 커밋 수입니다. 표와 CSV에는 모든 항목이 포함됩니다.')
    st.dataframe(table, width='stretch')
    st.download_button('현재 막대그래프 집계표 (.csv)', csv_bytes(table),
                       f'{column}_{start:%Y%m}_{end:%Y%m}.csv', 'text/csv')


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
    column_overview = render_data_quality(data, audit)
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
        exclude_outliers = st.checkbox('이상치가 있는 저장소 제외', value=False,
                                       help='월간 커밋이 비정상적으로 급증한 저장소를 전후 양쪽에서 제외합니다. 원본 데이터는 바뀌지 않습니다.')
        outlier_multiplier = st.slider('이상치 기준: 월 중앙값의 몇 배 초과', 3.0, 10.0, 5.0, step=.5,
                                       disabled=not exclude_outliers)
        zero_fill = st.checkbox('관측 사이의 빈 달을 커밋 0건으로 가정', value=False,
                                help='기본값은 빈 달이 있는 저장소를 제외합니다. 원본이 완전한 Git 히스토리라고 확인한 경우에만 켜세요.')
    filtered = data[data['repo'].isin(selected)].copy() if selected else data.copy()
    repo_selected_count = len(filtered)
    if exclude_merge:
        filtered = filtered[filtered['is_merge'] != '표시 있음']
    scoped, monthly, repos, empty_cells = compare(filtered, boundary, months, zero_fill)
    original_monthly = monthly.copy()
    original_repo_count, original_commit_count = len(repos), len(scoped)
    outliers = detect_monthly_outliers(scoped, outlier_multiplier)
    excluded_repos = outliers['저장소'].astype(str).unique().tolist() if exclude_outliers else []
    if excluded_repos:
        scoped, monthly, repos, empty_cells = compare(
            filtered.loc[~filtered['repo'].isin(excluded_repos)], boundary, months, zero_fill)
    removed_commits = original_commit_count - len(scoped)
    scope_steps = pd.DataFrame([
        {'단계': '저장소 선택', '처리 방법': '선택 저장소만 유지 (비우면 전체)',
         '처리 전 행 수': len(data), '제외 행 수': len(data) - repo_selected_count, '처리 후 행 수': repo_selected_count},
        {'단계': '병합 커밋', '처리 방법': '병합 커밋 제외' if exclude_merge else '제외하지 않음',
         '처리 전 행 수': repo_selected_count, '제외 행 수': repo_selected_count - len(filtered), '처리 후 행 수': len(filtered)},
        {'단계': '전후 비교 대상', '처리 방법': '기간·동일 저장소·빈 달 조건 적용 (결측치 제거와 별개인 분석 범위 선정)',
         '처리 전 행 수': len(filtered), '제외 행 수': len(filtered) - original_commit_count, '처리 후 행 수': original_commit_count},
        {'단계': '이상치 저장소', '처리 방법': '급증 후보 저장소를 전후 양쪽에서 제외' if exclude_outliers else '탐지만 하고 제외하지 않음',
         '처리 전 행 수': original_commit_count, '제외 행 수': removed_commits, '처리 후 행 수': len(scoped)}])
    st.subheader('분석 범위 · 이상치 처리 전후')
    st.dataframe(scope_steps, hide_index=True, width='stretch')
    st.caption(f'급증 후보 저장소 {outliers["저장소"].nunique():,}개 · 실제 제외 저장소 {len(excluded_repos):,}개. '
               '아래의 판정 기준과 제외 내역을 함께 확인하세요.')
    with st.expander('이상치 확인 · 제외 내역', expanded=exclude_outliers):
        st.caption(f'현재 비교 대상에서 관측 월이 12개 이상인 저장소만 검사합니다. 월간 커밋이 '
                   f'① 중앙값의 {outlier_multiplier:g}배, ② Q3 + 3 × IQR, ③ 100건을 모두 초과하면 급증 후보입니다. '
                   'IQR은 상위 25% 경계와 하위 25% 경계의 차이입니다. 빈 달은 탐지 계산에 넣지 않습니다.')
        st.caption('급증은 오류나 AI 사용의 증거가 아닙니다. 제외 기능을 켜면 해당 월만 지우거나 0건으로 바꾸지 않고, '
                   '그 저장소 전체를 이전·이후 비교에서 제외합니다. 모든 그래프·모델·보고서에 동일하게 적용됩니다.')
        if outliers.empty:
            st.write('현재 조건에서 급증 후보가 없습니다.')
        else:
            st.dataframe(outliers, hide_index=True, width="stretch")
        if exclude_outliers:
            st.write(f'제외 저장소 {len(excluded_repos):,}개 · 비교 기간에서 제외된 커밋 {removed_commits:,}건 · '
                     f'비교 저장소 {original_repo_count:,}개 → {len(repos):,}개')
            comparison = original_monthly[['커밋 수']].rename(columns={'커밋 수': '제외 전'})
            comparison['제외 후'] = monthly['커밋 수']
            st.dataframe(comparison, width="stretch")
    with st.expander('데이터와 전처리 내역', expanded=False):
        st.write({k: v for k, v in audit.items() if k != '원본 열 개요'})
        st.dataframe(sources, hide_index=True, width="stretch")
        st.write(f"관측 범위: {data['month'].min():%Y-%m} ~ {data['month'].max():%Y-%m} · 원본 저장소 {data['repo'].nunique():,}개")
        st.caption('원본의 모든 열을 읽습니다. 그래프와 내려받는 집계표에는 집계 결과를 표시합니다. 중복 키는 저장소+커밋 해시입니다. 작성 시각(UTC)을 사용합니다. 마지막 관측 월이 완전히 수집됐는지는 원본 설명으로 확인해야 합니다.')
        st.write('읽은 열:', ', '.join(data.columns))
        st.caption('AI 관련 미표시는 AI 미사용의 증거가 아닙니다. 작업 종류 누락은 미분류로 남겨 비율의 분모에 포함합니다.')
    st.caption(f'비교: {monthly.index.min():%Y-%m} ~ {(boundary - pd.DateOffset(months=1)):%Y-%m} / {boundary:%Y-%m} ~ {monthly.index.max():%Y-%m}. 각 {months}개월, 같은 저장소를 비교합니다.')
    st.caption('비교창 첫 달과 마지막 달까지 기록이 이어지는 저장소를 먼저 선택합니다. 빈 달 제외 모드는 매달 커밋이 있는 활발한 저장소에 치우칠 수 있습니다.')
    if repos.empty:
        st.warning('전후 비교 조건을 충족하는 저장소가 없습니다. 기간·저장소 선택을 바꾸거나 이상치 제외를 해제하세요. 아래 막대그래프에서는 별도 기간으로 탐색할 수 있습니다.')
        bar_data = filtered.loc[~filtered['repo'].isin(excluded_repos)]
        if not bar_data.empty:
            observed_monthly = bar_data.groupby('month', observed=True).size().rename('커밋 수').rename_axis('월').to_frame()
            render_activity(observed_monthly, boundary, comparable=False)
            render_composition(bar_data, boundary)
        return
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
        render_activity(monthly, boundary)
        st.subheader('저장소마다 변화가 같을까?')
        st.caption(f"증가 {(repos['차이'] > 0).sum()}개 · 감소 {(repos['차이'] < 0).sum()}개 · 동일 {(repos['차이'] == 0).sum()}개")
        st.dataframe(repos.sort_values('차이', ascending=False), width="stretch")
    with tabs[1]:
        render_composition(filtered.loc[~filtered['repo'].isin(excluded_repos)], boundary)
    scores, predictions, split = model_comparison(monthly, boundary)
    with tabs[2]:
        st.write(f'시간순 앞 {split}개월 학습 / 뒤 {len(monthly)-split}개월 평가. MAE는 월별 예측이 실제 값에서 평균 몇 건 벗어났는지를 뜻합니다.')
        st.dataframe(scores, hide_index=True, width="stretch")
        prediction_plot = predictions.rename_axis('평가 월').reset_index().melt(
            id_vars='평가 월', var_name='모델', value_name='커밋 수')
        st.plotly_chart(px.line(prediction_plot, x='평가 월', y='커밋 수', color='모델',
                                title='월별 커밋 활동량: 실제 값과 예측 비교',
                                labels={'평가 월': '평가 월 (UTC)', '커밋 수': '커밋 수 (건)'}), width="stretch")
        best_score = scores.iloc[0]
        st.write(f'발견: 평가 MAE가 가장 낮은 모델은 {best_score["모델"]}. '
                 f'근거: 평가 {len(predictions)}개월의 MAE {best_score["평가 MAE"]:,.2f}건, '
                 f'비교 모델 {len(scores)}개.')
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
- 이상치 저장소 제외: {exclude_outliers}
- 이상치 기준: 관측 월 12개 이상, 월간 커밋이 중앙값의 {outlier_multiplier:g}배 및 Q3 + 3 × IQR 및 100건 모두 초과
- 이상치 탐지 범위: 저장소·병합·기간·빈 달 조건을 적용한 비교 대상, 제외 전에 한 번 검사
- 이상치로 제외한 저장소: {', '.join(excluded_repos) if excluded_repos else '없음'}
- 이상치 저장소 제외로 빠진 비교 커밋: {removed_commits:,}건 (저장소 전체를 전후 양쪽에서 제외)

## 전처리
{cleaning_steps(audit).to_csv(index=False)}

## 분석 범위와 이상치 처리 전후
{scope_steps.to_csv(index=False)}

## 데이터 개요: 원본 {audit['원본 행 수']:,}행 × {audit['원본 열 수']}열
{column_overview.to_csv(index=False)}

## 월별 커밋 활동 그래프의 발견과 근거
월평균 이전 {before:,.2f}건 → 이후 {after:,.2f}건, 변화율 {delta_text}.

## 예측 그래프의 발견과 근거
평가 MAE 최저: {best['모델']}, {best['평가 MAE']:,.2f}건. 평가 기간 {len(predictions)}개월.

## 작업 종류별 전후 비교
{composition(scoped, 'commit_type').to_csv()}

## 해석의 범위
커밋 수는 개발 활동량을 나타내며 코드 품질이나 생산성을 직접 증명하지 않는다.
작업 종류 비율도 품질 점수가 아니다. AI 관련 라벨이 없어도 AI를 사용했을 수 있다.
시간 변화, 프로젝트 성장, 저장소 선정 편향을 통제하지 않았으므로 AI의 인과적 효과로 단정할 수 없다.
월별 관측 누락과 0건을 원본 데이터만으로 구분할 수 없다. 창 전체에서 활동한 저장소를 선택하는 데 따른 생존 편향이 있다.
평가 구간으로 모델을 선택했으며 별도 최종 시험 구간은 없다.
급증 후보는 데이터 오류로 확정된 것이 아니다. 이상치 제외 여부에 따라 결과가 달라질 수 있으므로 제외 전후 결과를 함께 확인해야 한다.
'''
        st.markdown(report.split('## 전처리')[0])
        st.download_button('결과 보고서 (.md)', report.encode('utf-8'), 'yulstar_report.md', 'text/markdown')
        tables = {'monthly.csv':monthly, 'repositories.csv':repos, 'commit_types.csv':composition(scoped,'commit_type'),
                  'ai_assisted.csv':composition(scoped,'ai_assisted'), 'models.csv':scores, 'predictions.csv':predictions,
                  'ai_tools.csv':composition(scoped,'ai_tool'), 'ai_models.csv':composition(scoped,'ai_model'),
                  'categories.csv':composition(scoped,'category'), 'coauthors.csv':composition(scoped,'n_coauthors'),
                  'source_files.csv':sources, 'preprocessing.csv':pd.DataFrame([{k:v for k,v in audit.items() if k != '원본 열 개요'}]),
                  'column_overview.csv':column_overview, 'cleaning_steps.csv':cleaning_steps(audit),
                  'scope_and_outlier_steps.csv':scope_steps,
                  'outlier_candidates.csv':outliers,
                  'excluded_repositories.csv':pd.DataFrame({'저장소': excluded_repos}),
                  'monthly_before_outlier_filter.csv':original_monthly}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for filename, frame in tables.items():
                archive.writestr(filename, csv_bytes(frame))
            archive.writestr('report.md', report)
        st.download_button('집계표 전체 (.zip)', buffer.getvalue(), 'yulstar_results.zip', 'application/zip')
        st.caption('CSV는 엑셀에서 한글이 깨지지 않도록 UTF-8 BOM으로 저장합니다. 원본 개인 식별 정보는 다운로드에 포함되지 않습니다.')


if __name__ == '__main__':
    main()
