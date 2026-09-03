# F1 Race Prediction Data Architecture

## Historical Coverage

Primary historical data coverage:

2014 onwards

## Prediction Philosophy

The prediction system must only use information that would have been
available at the prediction time.

Prediction stages must therefore be treated separately.

### Normal Weekend

1. Pre-FP1
2. After FP1
3. After FP2
4. After FP3
5. After Qualifying
6. Race result

### Sprint Weekend

1. Pre-FP1
2. After FP1
3. After Sprint Qualifying
4. After Sprint
5. After Main Qualifying
6. Main Race result

## Historical Data Categories

- Race results
- Qualifying results
- Practice results
- Sprint results
- Driver information
- Team information
- Circuit information
- Calendar information
- Weather
- Tyre compounds
- Driver form
- Team form
- Reliability
- Session timing
- Championship standings

## Tyre Representation

Tyres must be represented using the actual Pirelli compound
designation where applicable:

C1
C2
C3
C4
C5

Do not reduce tyre information to only:

Soft
Medium
Hard

## Important Data Rule

Practice sessions are input features.

Practice results must never become the prediction target.

The system must prevent future-session information from leaking into
earlier predictions.