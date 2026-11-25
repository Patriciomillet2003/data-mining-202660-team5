# Data dictionary

File: capstone_streamlit_app/data/team_dataset.csv

One row is one used multifunction printer that was still in stock on 20 October 2026. The table has 1,091 rows and 17 columns.

| Column | Type | Meaning | Values in the data |
|---|---|---|---|
| lot_id | text | Anonymous code of the purchase lot. Units of a lot arrived together. It is used to split training and test data by lot and is not a model input | LOT_001 to LOT_097; 97 lots |
| brand | category | Manufacturer | 10 brands. Most common: SHARP (339), CANON (203), KONICA MINOLTA (161) |
| line | category | Product line: colour (CO) or monochrome (MO), paper size (A3 or A4) and speed range in pages per minute | 9 levels: CO/A3/20-35 (265), MO/A4/20-70 (204), MO/A3/36-55 (173), MO/A3/20-35 (163), CO/A4/20-50 (118), CO/A3/36-60 (104), MO/A3/56-80 (45), CO/A3/61-90 (13), SIN LINEA (6) |
| warehouse | category | Warehouse that holds the unit: Mérida (MID), México (MEX), Monterrey (MTY) or Guadalajara (GDL) | MID (433), MEX (351), MTY (252), GDL (55) |
| appearance | category | Overall cosmetic condition | BUENO (932), EXCELENTE (158), REGULAR (1) |
| copy_quality | category | Quality of the printed copy | BUENO (916), EXCELENTE (173), REGULAR (2) |
| working | category | Whether the unit works | SI (1087), NO (4) |
| page_counter_k | number | Pages printed so far, in thousands | 1 to 1,615; median 140; 7 missing |
| landed_cost_mxn | number | Cost of the unit including freight, in Mexican pesos | 919 to 40,556; median 7,691 |
| acc_duplex | 0 or 1 | 1 when the accessories text of the unit mentions DUPLEX | 84% of units have it |
| acc_dsdf | 0 or 1 | 1 when the accessories text of the unit mentions DSDF | 65% of units have it |
| acc_fax | 0 or 1 | 1 when the accessories text of the unit mentions FAX | 60% of units have it |
| acc_radf | 0 or 1 | 1 when the accessories text of the unit mentions RADF | 35% of units have it |
| acc_finisher | 0 or 1 | 1 when the accessories text of the unit mentions FINISHER | 33% of units have it |
| acc_usend | 0 or 1 | 1 when the accessories text of the unit mentions U-SEND | 5% of units have it |
| days_in_stock | number | Days since the unit arrived | 8 to 1,492; median 47 |
| slow_mover_flag | 0 or 1 | Target. 1 when days_in_stock is above 120 | 281 units are slow movers (25.8%) |

## Notes

- days_in_stock defines the target, so no model receives it as an input. The dashboard uses it for the 120-day discount rule.
- A page counter of 0 on a used machine is a reading nobody took. It was stored as missing, and the pipeline fills it with the median of the training data (137).
- The columns that identify clients, sellers, suppliers, invoices, customs entries and serial numbers were removed before the data reached the repository. The arrival date, the exchange rate, the reservation flag and the transfer history were removed too, because they reveal how old the unit is.
- Levels with very few units (for example LANIER or SIN LINEA) are pooled into one "infrequent" column when the model matrix is built.
