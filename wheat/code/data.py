"""
data.py -- Shao et al. (2026) Logi-PINN 논문과 동일한 전처리/데이터 분할을 재현한다.

논문 설정 (Section 3.1):
  - 파종일 기준 정렬된 285일 시계열, 앞 115일 제거 -> n_t = 170일
  - 각 연도별 첫 측정 이전 시점은 해당 시퀀스의 최소 높이값으로 채움
  - 4개 연도(2018, 2019, 2021, 2022) 모두에 존재하는 19개 유전자형
  - 학습 2년 / 검증 1년 / 테스트 1년 (2:1:1), 6가지 분할 조합
  - 학습 시에는 replicate(plot)를 독립 샘플로 사용,
    평가 시에는 같은 (genotype, year)의 replicate를 평균한 곡선으로 RMSE 계산

이 모듈은 torch에 의존하지 않는다 (numpy/pandas만 사용). 따라서 모델 없이도
전처리 파이프라인만 따로 검증할 수 있다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 파일 경로 해석
# --------------------------------------------------------------------------- #
# 파일명만 넘겨도 찾을 수 있도록 아래 순서로 탐색한다.
#   1) 넘겨받은 경로 그대로 (절대경로이거나 현재 디렉터리 기준 상대경로)
#   2) 환경변수 PLANT_DATA_DIR 에 나열된 디렉터리들 (os.pathsep 로 구분)
#   3) 현재 작업 디렉터리, 이 스크립트의 디렉터리와 그 부모
# 각 디렉터리는 바로 아래 하위 폴더까지 함께 훑는다.
#
# 환경변수 사용 예:
#   Windows     : set PLANT_DATA_DIR=C:\PlantNeuralODE
#                 -> C:\PlantNeuralODE\processed_data, ...\temporary 를 모두 찾음
#   Linux/macOS : export PLANT_DATA_DIR=~/PlantNeuralODE
_HERE = Path(__file__).resolve().parent
_DEFAULT_SEARCH_DIRS = (Path.cwd(), _HERE, _HERE.parent)

DEFAULT_DATA_FILE = "align_height_env_same_length.csv"
DEFAULT_KINSHIP_FILE = "kinship_matrix_astle.csv"


def search_dirs() -> list[Path]:
    """탐색 대상 디렉터리 목록. 환경변수 PLANT_DATA_DIR 가 우선.

    각 디렉터리의 **바로 아래 하위 폴더까지** 함께 훑는다. 그래서
    C:\\PlantNeuralODE 하나만 지정해도 그 안의 processed_data, temporary 를
    모두 찾는다. (그 아래 손자 폴더까지 파고들지는 않는다.)

    단, 드라이브 루트(C:\\, /)는 하위 폴더를 훑지 않는다. 코드가 C:\\PlantNeuralODE
    에 있을 때 부모가 C:\\ 가 되는데, 여기서 하위를 훑으면 Program Files 나
    System Volume Information 까지 뒤지게 되어 느리고 무의미하다.
    """
    roots = []
    env = os.environ.get("PLANT_DATA_DIR", "")
    for part in env.split(os.pathsep):
        part = part.strip()
        if part:
            roots.append(Path(part).expanduser())
    roots.extend(_DEFAULT_SEARCH_DIRS)

    out, seen = [], set()
    for d in roots:
        cands = [d] if _is_root(d) else [d] + _child_dirs(d)
        for cand in cands:
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
    return out


def _is_root(d: Path) -> bool:
    """드라이브 루트(C:\\) 또는 파일시스템 루트(/) 인가."""
    return d.resolve().parent == d.resolve()


def _child_dirs(d: Path) -> list[Path]:
    """바로 아래 하위 디렉터리들 (숨김 폴더와 __pycache__ 는 제외)."""
    try:
        return sorted(
            c for c in d.iterdir()
            if c.is_dir() and not c.name.startswith(".") and c.name != "__pycache__"
        )
    except (OSError, PermissionError):
        return []


def resolve_path(path, default_name: str | None = None) -> str:
    """파일을 찾아 절대경로로 돌려준다. 못 찾으면 탐색한 위치를 알려주고 실패."""
    name = str(path) if path else default_name
    if name is None:
        raise ValueError("파일 경로가 주어지지 않았습니다.")

    p = Path(name).expanduser()
    if p.is_file():
        return str(p.resolve())
    if not p.is_absolute():
        for d in search_dirs():
            cand = d / p
            if cand.is_file():
                return str(cand.resolve())

    dirs = search_dirs()
    shown = [str(d / Path(name).name) for d in dirs[:8]]
    if len(dirs) > 8:
        shown.append(f"... 외 {len(dirs) - 8}곳")
    tried = "\n  ".join(shown)
    raise FileNotFoundError(
        f"'{name}' 을(를) 찾지 못했습니다. 확인한 위치:\n  {p}\n  {tried}\n"
        f"--data / --kinship 로 경로를 직접 지정하거나, 환경변수 PLANT_DATA_DIR 에 "
        f"데이터 디렉터리를 넣어 두세요."
    )

DEFAULT_YEARS = (2018, 2019, 2021, 2022)
FULL_SEQ_LEN = 285          # 파종일 정렬 후 전체 길이
DEFAULT_START_DAY = 115     # 논문에서 잘라내는 초기 구간
DEFAULT_ENV_COLS = ("Air_temperature_2_m",)

DAY_COL = "day_after_start_measure"
YEAR_COL = "year_site.harvest_year"
GENO_COL = "genotype.id"
PLOT_COL = "plot.UID"
VALUE_COL = "value"


# --------------------------------------------------------------------------- #
# 컨테이너
# --------------------------------------------------------------------------- #
@dataclass
class SequenceSet:
    """길이가 동일한 시퀀스들의 묶음.

    y     : [N, T]     식물 높이 (m). 관측이 없는 지점은 0으로 채워져 있고 mask=0.
    mask  : [N, T]     손실 계산에 사용할 지점 = 1
    env   : [N, T, E]  환경 변수 (기본: 2 m 기온)
    s     : [N, T]     정규화된 열시간(누적 GDD). 기준 상수로 나눠 대략 0~1.
    ds    : [N, T]     ds/dτ. 그날의 발육 진행 속도 (추운 날 ≈ 0).
    g_idx : [N]        유전자형 인덱스 (0..n_g-1)
    year  : [N]
    plot  : [N]        plot.UID (평균화된 시퀀스는 'avg')
    """

    y: np.ndarray
    mask: np.ndarray
    env: np.ndarray
    s: np.ndarray
    ds: np.ndarray
    g_idx: np.ndarray
    year: np.ndarray
    plot: np.ndarray

    def __len__(self) -> int:
        return self.y.shape[0]

    @property
    def n_t(self) -> int:
        return self.y.shape[1]

    def subset(self, idx) -> "SequenceSet":
        idx = np.asarray(idx)
        return SequenceSet(
            y=self.y[idx],
            mask=self.mask[idx],
            env=self.env[idx],
            s=self.s[idx],
            ds=self.ds[idx],
            g_idx=self.g_idx[idx],
            year=self.year[idx],
            plot=self.plot[idx],
        )

    def select_years(self, years) -> "SequenceSet":
        years = np.atleast_1d(np.asarray(years))
        return self.subset(np.isin(self.year, years))


@dataclass
class Dataset:
    train: SequenceSet          # replicate 단위 (학습용)
    val: SequenceSet            # replicate 평균 (평가용)
    test: SequenceSet           # replicate 평균 (평가용)
    train_eval: SequenceSet     # replicate 평균 (학습년도 평가용)
    genotypes: list             # 원본 genotype.id 목록 (인덱스 순서)
    kinship: np.ndarray         # [n_g, n_g]
    env_mean: np.ndarray        # [E]
    env_std: np.ndarray         # [E]
    gdd_scale: float            # 열시간 정규화 상수 (학습 연도 평균 총 GDD)
    train_years: tuple
    val_year: int
    test_year: int


# --------------------------------------------------------------------------- #
# 입력 파일 읽기
# --------------------------------------------------------------------------- #
def load_kinship(path=None):
    """kinship_matrix_astle.csv를 읽어 (genotype 목록, kinship 행렬)을 돌려준다."""
    path = resolve_path(path, DEFAULT_KINSHIP_FILE)
    k = pd.read_csv(path, header=0, index_col=0)
    genotypes = [int(g) for g in k.index]
    kin = k.to_numpy(dtype=np.float64)
    if kin.shape[0] != kin.shape[1]:
        raise ValueError(f"kinship 행렬이 정사각이 아닙니다: {kin.shape}")
    return genotypes, kin


def _build_daily_env(df: pd.DataFrame, years, env_cols, start_day: int, n_t: int,
                     add_gdd: bool = False, gdd_base: float = 0.0):
    """연도별 일 단위 환경 시계열을 만든다. -> dict[year] -> [n_t, E]

    align_height_env_same_length.csv 는 환경 변수가 '표현형 측정일에만' 기록되어
    있으므로(논문의 same_length 설정), 남는 날짜는 선형 보간으로 채운다.
    전체 길이 파일(align_height_env.csv)을 쓰면 보간이 사실상 항등이 된다.
    """
    days = np.arange(start_day, start_day + n_t)
    out = {}
    gdd_raw = {}
    for year in years:
        dy = df[df[YEAR_COL] == year]
        cols = []
        for col in env_cols:
            sub = dy.dropna(subset=[col])
            if len(sub) == 0:
                raise ValueError(f"{year}년에 '{col}' 값이 하나도 없습니다.")
            per_day = sub.groupby(DAY_COL)[col].mean().sort_index()
            # np.interp 는 범위 밖에서 양 끝 값으로 자동 확장된다.
            cols.append(np.interp(days, per_day.index.to_numpy(), per_day.to_numpy()))
        env = np.stack(cols, axis=-1)  # [n_t, E]

        # 열시간(thermal time). 일별 GDD = max(평균기온 − 기준온도, 0),
        # 누적하면 그 해의 발육 진도를 나타낸다. 정규화 상수는 make_dataset 에서
        # 학습 연도만 보고 결정하므로 여기서는 원시값을 넘긴다.
        daily_gdd = np.clip(env[:, 0] - gdd_base, 0.0, None)
        # 사다리꼴 누적. RK4 가 ds 를 적분한 결과와 s 가 정확히 일치하도록
        # (단순 cumsum 은 직사각형 합이라 0.4% 정도 어긋난다).
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (daily_gdd[:-1] + daily_gdd[1:]))])
        gdd_raw[year] = (cum, daily_gdd)

        if add_gdd:
            # 누적 생육도일(GDD). 창(115~284일) 안에서만 누적한다.
            # 연도 간 기온 조건의 차이를 명시적으로 넘겨주는 고전적 작물모형 특징.
            gdd = np.cumsum(np.clip(env[:, 0] - gdd_base, 0.0, None))
            env = np.concatenate([env, gdd[:, None]], axis=-1)
        out[year] = env
    return out, gdd_raw


def build_sequences(
    data_path=None,
    kinship_path=None,
    env_cols=DEFAULT_ENV_COLS,
    start_day: int = DEFAULT_START_DAY,
    n_t: int = FULL_SEQ_LEN - DEFAULT_START_DAY,
    years=DEFAULT_YEARS,
    fill_in_na_at_start: bool = True,
    genotypes=None,
    add_gdd: bool = False,
    gdd_base: float = 0.0,
):
    """CSV -> SequenceSet (replicate 단위) 변환.

    fill_in_na_at_start=True 이면 논문과 동일하게 각 시퀀스의 첫 관측 이전 시점을
    그 시퀀스의 최소 높이값으로 채우고, 해당 지점도 관측(mask=1)으로 취급한다.
    """
    data_path = resolve_path(data_path, DEFAULT_DATA_FILE)
    kinship_path = resolve_path(kinship_path, DEFAULT_KINSHIP_FILE)

    env_cols = list(env_cols)
    if genotypes is None:
        genotypes, kin = load_kinship(kinship_path)
    else:
        all_g, kin_full = load_kinship(kinship_path)
        genotypes = [int(g) for g in genotypes]
        unknown = [g for g in genotypes if g not in all_g]
        if unknown:
            raise ValueError(
                f"kinship 행렬에 없는 유전자형입니다: {unknown}\n"
                f"사용 가능한 유전자형: {all_g}"
            )
        pos = [all_g.index(g) for g in genotypes]
        kin = kin_full[np.ix_(pos, pos)]

    usecols = [DAY_COL, YEAR_COL, GENO_COL, PLOT_COL, VALUE_COL] + env_cols
    df = pd.read_csv(data_path, header=0, index_col=0, low_memory=False)
    missing = [c for c in usecols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 다음 컬럼이 없습니다: {missing}")
    df = df[usecols]
    df = df[df[GENO_COL].isin(genotypes) & df[YEAR_COL].isin(years)]

    g_lookup = {g: i for i, g in enumerate(genotypes)}
    env_by_year, gdd_by_year = _build_daily_env(
        df, years, env_cols, start_day, n_t, add_gdd, gdd_base)

    days = np.arange(start_day, start_day + n_t)
    ys, masks, envs, ss, dss, gs, yrs, plots = [], [], [], [], [], [], [], []

    for (plot, year, geno), grp in df.groupby([PLOT_COL, YEAR_COL, GENO_COL], sort=True):
        s = grp.set_index(DAY_COL)[VALUE_COL]
        s = s[~s.index.duplicated(keep="first")].reindex(days)
        vals = s.to_numpy(dtype=np.float64)
        obs = ~np.isnan(vals)
        if obs.sum() == 0:
            continue

        y = np.zeros(n_t, dtype=np.float64)
        mask = np.zeros(n_t, dtype=np.float64)
        y[obs] = vals[obs]
        mask[obs] = 1.0

        if fill_in_na_at_start:
            first = int(np.argmax(obs))
            if first > 0:
                y[:first] = np.nanmin(vals)
                mask[:first] = 1.0

        cum, daily = gdd_by_year[year]
        ys.append(y)
        masks.append(mask)
        envs.append(env_by_year[year])
        ss.append(cum)
        dss.append(daily * (n_t - 1))   # dt/dτ = n_t-1 이므로 ds/dτ = daily*(n_t-1)
        gs.append(g_lookup[geno])
        yrs.append(year)
        plots.append(plot)

    seq = SequenceSet(
        y=np.stack(ys),
        mask=np.stack(masks),
        env=np.stack(envs),
        s=np.stack(ss),
        ds=np.stack(dss),
        g_idx=np.asarray(gs, dtype=np.int64),
        year=np.asarray(yrs, dtype=np.int64),
        plot=np.asarray(plots, dtype=object),
    )
    return seq, genotypes, kin


# --------------------------------------------------------------------------- #
# replicate 평균 / 분할
# --------------------------------------------------------------------------- #
def average_replicates(seq: SequenceSet) -> SequenceSet:
    """같은 (genotype, year)의 replicate를 시점별로 평균한다 (논문의 평가 방식).

    한 시점에서 관측이 있는 replicate만 평균하고, 하나라도 있으면 mask=1.
    """
    keys = sorted({(int(g), int(y)) for g, y in zip(seq.g_idx, seq.year)})
    ys, masks, envs, ss, dss, gs, yrs, plots = [], [], [], [], [], [], [], []
    for g, year in keys:
        idx = np.where((seq.g_idx == g) & (seq.year == year))[0]
        m = seq.mask[idx]                       # [R, T]
        denom = m.sum(axis=0)
        num = (seq.y[idx] * m).sum(axis=0)
        y = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
        ys.append(y)
        masks.append((denom > 0).astype(np.float64))
        envs.append(seq.env[idx[0]])
        ss.append(seq.s[idx[0]])
        dss.append(seq.ds[idx[0]])
        gs.append(g)
        yrs.append(year)
        plots.append("avg")
    return SequenceSet(
        y=np.stack(ys),
        mask=np.stack(masks),
        env=np.stack(envs),
        s=np.stack(ss),
        ds=np.stack(dss),
        g_idx=np.asarray(gs, dtype=np.int64),
        year=np.asarray(yrs, dtype=np.int64),
        plot=np.asarray(plots, dtype=object),
    )


def all_year_splits(years=DEFAULT_YEARS):
    """논문과 같이 학습 2년의 모든 조합 (6가지)을 만든다.

    남는 두 해 중 앞선 해를 검증, 나중 해를 테스트로 둔다.
    split 0 = train (2018, 2019) / val 2021 / test 2022  <- 논문 본문 기준 설정
    """
    years = tuple(sorted(years))
    splits = []
    for train in combinations(years, 2):
        rest = sorted(set(years) - set(train))
        splits.append({"train_years": train, "val_year": rest[0], "test_year": rest[1]})
    # 논문이 집중한 (2018, 2019) 학습 조합을 맨 앞으로
    splits.sort(key=lambda s: (s["train_years"] != (2018, 2019), s["train_years"]))
    return splits


def standardize_env(seq: SequenceSet, mean: np.ndarray, std: np.ndarray) -> SequenceSet:
    return replace(seq, env=(seq.env - mean) / std)


def normalize_thermal_time(seq: SequenceSet, scale: float) -> SequenceSet:
    """누적 GDD 를 공통 상수로 나눠 열시간 s 를 만든다.

    연도별 총합으로 나누면 모든 해가 0→1 이 되어 '어느 해가 더 따뜻했는가' 정보가
    사라진다. 그래서 **학습 연도 평균 총 GDD** 라는 단일 상수로 나눈다. 따뜻한 해는
    s(끝) > 1, 추운 해는 < 1 이 되어 총 적산량 차이가 보존된다.
    """
    return replace(seq, s=seq.s / scale, ds=seq.ds / scale)


def make_dataset(
    data_path=None,
    kinship_path=None,
    split: int = 0,
    env_cols=DEFAULT_ENV_COLS,
    start_day: int = DEFAULT_START_DAY,
    n_t: int = FULL_SEQ_LEN - DEFAULT_START_DAY,
    fill_in_na_at_start: bool = True,
    val_year: int | None = None,
    test_year: int | None = None,
    add_gdd: bool = False,
    gdd_base: float = 0.0,
    genotypes=None,
) -> Dataset:
    """전처리 + 연도 분할 + 환경변수 표준화를 한 번에 수행한다.

    genotypes 에 목록을 주면 그 유전자형만 사용한다 (None 이면 19개 전부).
    예: genotypes=[106] -> 논문 Table 4 의 단일 유전자형 설정
    """
    seq, genotypes, kin = build_sequences(
        data_path,
        kinship_path,
        env_cols=env_cols,
        start_day=start_day,
        n_t=n_t,
        fill_in_na_at_start=fill_in_na_at_start,
        add_gdd=add_gdd,
        gdd_base=gdd_base,
        genotypes=genotypes,
    )

    cfg = all_year_splits()[split]
    train_years = cfg["train_years"]
    val_year = cfg["val_year"] if val_year is None else val_year
    test_year = cfg["test_year"] if test_year is None else test_year
    if val_year == test_year or val_year in train_years or test_year in train_years:
        raise ValueError("학습/검증/테스트 연도가 겹칩니다.")

    train = seq.select_years(train_years)
    env_mean = train.env.reshape(-1, train.env.shape[-1]).mean(axis=0)
    env_std = train.env.reshape(-1, train.env.shape[-1]).std(axis=0)
    env_std[env_std < 1e-8] = 1.0

    # 열시간 정규화 상수: 학습 연도의 평균 총 누적 GDD (검증/테스트 정보 미사용)
    gdd_scale = float(np.mean([train.s[i, -1] for i in range(len(train))]))

    train = normalize_thermal_time(standardize_env(train, env_mean, env_std), gdd_scale)
    val = normalize_thermal_time(
        standardize_env(average_replicates(seq.select_years(val_year)), env_mean, env_std),
        gdd_scale)
    test = normalize_thermal_time(
        standardize_env(average_replicates(seq.select_years(test_year)), env_mean, env_std),
        gdd_scale)
    train_eval = average_replicates(train)

    return Dataset(
        train=train,
        val=val,
        test=test,
        train_eval=train_eval,
        genotypes=genotypes,
        kinship=kin,
        env_mean=env_mean,
        env_std=env_std,
        gdd_scale=gdd_scale,
        train_years=train_years,
        val_year=val_year,
        test_year=test_year,
    )


# --------------------------------------------------------------------------- #
# 지표
# --------------------------------------------------------------------------- #
def masked_rmse(pred: np.ndarray, y: np.ndarray, mask: np.ndarray, per_sample: bool = False):
    """논문 Eq.(3)와 동일하게, 시퀀스별 RMSE를 구한 뒤 시퀀스에 대해 평균한다."""
    err = ((pred - y) ** 2) * mask
    denom = np.clip(mask.sum(axis=-1), 1e-8, None)
    rmse_per_seq = np.sqrt(err.sum(axis=-1) / denom)
    return rmse_per_seq if per_sample else float(rmse_per_seq.mean())


if __name__ == "__main__":  # 간단한 자체 점검
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None,
                   help=f"기본값: {DEFAULT_DATA_FILE} (PLANT_DATA_DIR 등에서 탐색)")
    p.add_argument("--kinship", default=None,
                   help=f"기본값: {DEFAULT_KINSHIP_FILE} (PLANT_DATA_DIR 등에서 탐색)")
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--add_gdd", action="store_true")
    p.add_argument("--genotypes", type=int, nargs="*", default=None,
                   help="사용할 유전자형 목록. 비우면 19개 전부.")
    a = p.parse_args()

    ds = make_dataset(a.data, a.kinship, split=a.split, add_gdd=a.add_gdd,
                      genotypes=a.genotypes)
    print(f"genotypes ({len(ds.genotypes)}): {ds.genotypes}")
    print(f"split: train {ds.train_years} / val {ds.val_year} / test {ds.test_year}")
    for name in ["train", "train_eval", "val", "test"]:
        s = getattr(ds, name)
        print(f"  {name:10s} N={len(s):4d}  T={s.n_t}  "
              f"obs/seq={s.mask.sum(1).mean():.1f}  y max={s.y.max():.3f}  "
              f"s(끝)={s.s[:, -1].mean():.3f}")
    print(f"env mean/std: {ds.env_mean}, {ds.env_std}")
    print(f"열시간 정규화 상수(학습 연도 평균 총 GDD): {ds.gdd_scale:.0f} °C·day")