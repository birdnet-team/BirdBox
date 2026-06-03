


## Ground-Truth Labels CSV

Evaluation scripts expect a CSV with these columns:

```text
Filename,Start Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Species eBird Code
```

Conventions:

- time in seconds
- frequencies in Hz
- species as eBird code
- filename matching is normalized by base name (extension removed)
