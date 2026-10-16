# WALL-E EVE Signal -- Mission Report

## 1. Mission Summary

| Field            | Value                          |
|------------------|--------------------------------|
| Config Name      | `sample_hard`          |
| Grid Size        | 15 x 15                         |
| Seed             | 99        |
| Total Steps      | 42                        |
| Mission Result   | [FOUND] EVE Located                       |
| Terminal Event   | `found_eve:W|terminal:found_eve`                      |
| Replans          | 6                 |

## 2. Evidence Found

| Step | Position (x,y) | Clue Direction | Clue Strength |
|------|----------------|----------------|---------------|
| 11 | (5, 3) | N | 0.930 |
| 16 | (9, 3) | N | 0.570 |
| 21 | (12, 2) | SW | 0.555 |
| 26 | (14, 2) | SW | 0.454 |
| 31 | (12, 4) | SE | 0.972 |
| 36 | (13, 7) | N | 0.437 |

## 3. Inferred EVE Location and Status

| Field                | Value                |
|----------------------|----------------------|
| Most Likely Cell     | (12, 11) -- CONFIRMED       |
| Cell Probability     | 100.0%           |
| Confidence Score     | 100.0%     |
| Belief Entropy       | 0.0000        |
| EVE Status Estimate  | likely active [confirmed]         |

### Confirmed Location

| Rank | Position (x,y) | Probability |
|------|----------------|-------------|
| 1    | (12, 11)   | 100.00%     |

## 4. Resources Used vs Remaining

| Resource      | Used       | Remaining  | Limit      |
|---------------|------------|------------|------------|
| Battery       |        171 |         49 |        220 |
| Steps / Time  |         42 |         58 |        100 |

## 5. Recommended Next Step for WALL-E

> **EVE confirmed at (12, 11) -- proceed to that cell**

EVE was physically located at (12, 11) during this mission. WALL-E should navigate directly there.

EVE operational status: **likely active [confirmed]**.
