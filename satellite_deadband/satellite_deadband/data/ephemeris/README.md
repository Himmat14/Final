# Real ephemerides (optional)

Put CCSDS OEM ephemeris files (`*.oem` or `*.txt`) here, for example Starlink's public ephemerides from
space-track.org (free account needed; "Public Files" -> Starlink). `python main.py --stages step11` then reads
them (`deadband/observation.read_oem`), plots their mean SMA (`step11_real_ephemeris.png`) and can run the same
burn tests on them. No files are shipped with the code.
