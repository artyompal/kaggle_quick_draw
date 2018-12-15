# Google Quick Draw Challenge

## Train/validation split
There are 49 millions of samples. I reserve 1M samples for validation (every 30th sample).
The rest of samples is split into blocks of 10k samples per class. One block is fed at a single epoch.

