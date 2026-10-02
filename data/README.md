# Data

원본 데이터는 용량 문제(배치당 2 ~ 3GB)로 저장소에 포함하지 않는다.

## 다운로드

MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle?resource=download

아래 3개 파일을 받아 이 폴더(`data/`)에 둔다.

| Batch | 파일 | 용도 |
|---|---|---|
| Batch 1 | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | EDA, 학습 |
| Batch 2 | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | EDA, 테스트 |
| Batch 3 | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | EDA |

## 전처리

```bash
python -m src.preprocess
```

`data/processed/`에 아래 캐시가 생성된다.

- `cells.parquet` : 셀 단위 메타데이터 (batch, 충전 정책, cycle_life, 제외 사유)
- `summary.parquet` : 사이클 단위 요약값 (QD, QC, IR, 온도, 충전 시간)
- `curves.npz` : 초기 1 ~ 100 사이클의 Qdlin / Tdlin 곡선

## 셀 제외 기준

| 사유 | 기준 | 대상 |
|---|---|---|
| `censored` | EOL(0.88Ah) 도달 전 실험 종료 | Batch 1 의 10개 셀 |
| `no_cycle_life` | cycle_life 없음 (VarCharge / SLOWCYCLE 실험 셀, 미도달 셀) | Batch 2 의 8개, Batch 3 의 2개 |

최종 사용 셀 : Batch 1 36개, Batch 2 39개, Batch 3 44개
