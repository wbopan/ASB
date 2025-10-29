"""Minimal airport metadata to satisfy outlines' type requirements."""

# The original `pyairports` package distributes a comprehensive dataset of airports,
# but that project does not publish importable modules on PyPI at the moment.  The
# outlines library (pulled in by vLLM) only needs the IATA codes located at index 3
# of each record.  To keep vLLM functional without the upstream dependency, we ship
# a small curated subset of major airports.

AIRPORT_LIST = [
    ("1", "Hartsfield-Jackson Atlanta International", "Atlanta", "ATL", "US"),
    ("2", "Los Angeles International", "Los Angeles", "LAX", "US"),
    ("3", "O'Hare International", "Chicago", "ORD", "US"),
    ("4", "Haneda Airport", "Tokyo", "HND", "JP"),
    ("5", "Heathrow Airport", "London", "LHR", "GB"),
    ("6", "Dallas/Fort Worth International", "Dallas", "DFW", "US"),
    ("7", "Denver International", "Denver", "DEN", "US"),
    ("8", "John F. Kennedy International", "New York", "JFK", "US"),
    ("9", "San Francisco International", "San Francisco", "SFO", "US"),
    ("10", "Seattle-Tacoma International", "Seattle", "SEA", "US"),
    ("11", "Singapore Changi Airport", "Singapore", "SIN", "SG"),
    ("12", "Incheon International", "Seoul", "ICN", "KR"),
    ("13", "Frankfurt Airport", "Frankfurt", "FRA", "DE"),
    ("14", "Charles de Gaulle Airport", "Paris", "CDG", "FR"),
    ("15", "Sydney Kingsford Smith", "Sydney", "SYD", "AU"),
]

