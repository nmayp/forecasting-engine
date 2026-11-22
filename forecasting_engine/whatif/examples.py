"""One-click example requests for the five what-if mechanisms, and the scenario keys each one reads."""
from __future__ import annotations

EXAMPLES: dict[str, dict] = {
    "heat_load": dict(
        title="A 7-day heat wave and Spain's and Italy's electricity demand",
        description="Population-weighted daily mean temperatures of 26-32 degC for a week in Spain and Italy: the electricity demand the temperature adds against a normal week, per day and "
                    "local hour, with p10-p90 (screening: size good to about +-20 % on five past heat waves).",
        request={"area": {"country": ["ES", "IT"]},
                 "scenario": {"kind": "custom", "start": "2026-07-14", "tmean_c": [26, 28, 30, 31, 32, 30, 27]}}, network=False),
    "strike_air": dict(
        title="Three Belgian national strike days: flights lost",
        description="Brussels cancels all departures and half of arrivals on three strike days (announced share 0.75): the share of Belgium's flights lost per day with p10-p90 "
                    "(experimental, relative effect only).",
        request={"area": {"country": ["BE"]}, "scenario": {"kind": "custom", "start": "2026-11-24", "days": 3, "intensity": 0.75}}, network=False),
    "rates_credit": dict(
        title="ECB cuts to 1.5 % over a year: Germany's and Spain's cost of borrowing",
        description="A path of the ECB deposit rate (2.25 % falling to 1.5 %): the cost of new loans to firms and for house purchase in Germany and Spain over 12 months, with p10-p90 "
                    "(screening: direction and size back-tested, timing is 4-6 months late, no real-side effects).",
        request={"area": {"country": ["DE", "ES"]},
                 "scenario": {"kind": "custom", "rate_path_pct": [2.25, 2.0, 2.0, 1.75, 1.75, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]}, "horizon_h": 8766}, network=False),
    "oil_pump_hicp": dict(
        title="Wholesale diesel and petrol +25 %: pump prices and inflation in Germany, Spain and France",
        description="The wholesale product price rises 25 % over five weeks and stays: the diesel and petrol pump price per week and the fuel component of consumer prices over four months "
                    "(screening, a given-the-path what-if; electricity-gas is only a direction indicator).",
        request={"area": {"country": ["DE", "ES", "FR"]},
                 "scenario": {"kind": "custom", "product_change_pct": [0, 5, 10, 15, 20, 25, 25, 25, 25, 25, 25, 25], "gas_delta_eur_mwh": [0, 10, 10, 10]}, "horizon_h": 2016}, network=False),
    "choke_freight": dict(
        title="Suez container transits down 55 % for 26 weeks: Asia-Europe freight",
        description="Container transits through Suez fall by 55 % and stay down: the Shanghai-Rotterdam container freight rate as a multiple of the pre-onset level, with a sensitivity range "
                    "(experimental: direction and order of magnitude only).",
        request={"area": {"lat": 30.0, "lon": 32.55, "radius_km": 30}, "scenario": {"kind": "custom", "lane": "suez", "transit_cut_pct": 55, "weeks": 26}, "horizon_h": 4368}, network=False),}

SCENARIO_KEYS: dict[str, dict] = {
    "heat_load": {"tmean_c": {"unit": "degC", "range": [-40, 55], "doc": "custom: daily population-weighted mean temperature, a list or {country: list}, up to 120 days"},
                  "anomaly_c": {"unit": "degC", "range": [-15, 25], "doc": "anomaly: degrees above the day-of-year normal, a number (held `days` days) or a list or {country: ..}"},
                  "start": {"unit": "date"}, "reference_c": {"unit": "degC", "doc": "optional reference path to compare with (default: the normal)"},
                  "baseline_mw": {"unit": "MW", "doc": "optional daily mean load of your own: gives the excess in MW"},
                  "wind_mw": {"unit": "MW", "doc": "optional daily mean wind generation (number, list or {country: list}): CONTEXT only, not used by the demand mechanism; with baseline_mw gives an arithmetic residual load"},
                  "solar_mw": {"unit": "MW", "doc": "optional daily mean solar generation: context only, as wind_mw"}},
    "rates_credit": {"rate_path_pct": {"unit": "% per year", "range": [-1, 10], "doc": "monthly mean ECB deposit facility rate from the origin month (up to 12 values, held after the last)"},
                     "steps": {"unit": "list", "doc": "alternative: [{month: YYYY-MM, rate_pct | change_pp}]"}, "months": {"unit": "months", "range": [1, 12]},
                     "pricing": {"unit": "id", "doc": "announced (priced by the market, default) | surprise (not priced)"}},
    "oil_pump_hicp": {"product_change_pct": {"unit": "%", "range": [-90, 500], "doc": "cumulative change of the wholesale product price vs now, per week (up to 26) or a number held"},
                      "brent_usd_bbl": {"unit": "USD/bbl", "doc": "alternative: Brent levels per week; the product moves by the same EUR/l"},
                      "diesel_change_pct": {"unit": "%", "range": [-90, 500], "doc": "optional: the diesel wholesale price path on its own (cumulative % vs now, per week); replaces the common path for diesel"},
                      "gasoline_change_pct": {"unit": "%", "range": [-90, 500], "doc": "optional: the same for petrol (gasoline)"},
                      "crack_change_usd_bbl": {"unit": "USD/bbl", "range": [-150, 300], "doc": "optional: change of the product margin over Brent (crack spread) vs now, per week or a number held; added to the Brent / product path (or alone, Brent unchanged)"},
                      "diesel_crack_change_usd_bbl": {"unit": "USD/bbl", "doc": "optional: the crack change for diesel only"}, "gasoline_crack_change_usd_bbl": {"unit": "USD/bbl", "doc": "optional: the crack change for petrol only"},
                      "pump_change_pct": {"unit": "%", "doc": "alternative: the pump-price path itself (skips the pass-through mechanism)"},
                      "gas_delta_eur_mwh": {"unit": "EUR/MWh", "doc": "cumulative change of the gas price vs now, per month (up to 6)"},
                      "elec_delta_eur_mwh": {"unit": "EUR/MWh", "doc": "same for electricity (8 countries have an electricity term)"}, "months": {"unit": "months", "range": [1, 6]}},
    "choke_freight": {"lane": {"unit": "id", "doc": "suez | panama"}, "transit_cut_pct": {"unit": "%", "range": [0, 100], "doc": "container transits lost vs the pre-onset mean, per week or a number held"},
                      "weeks": {"unit": "weeks", "range": [1, 52]}, "index": {"unit": "id", "doc": "wci_rtm (Shanghai-Rotterdam) | wci | scfi"}},
    "strike_air": {"start": {"unit": "date"}, "days": {"unit": "days", "range": [1, 60]}, "dates": {"unit": "list of dates", "doc": "explicit strike days"},
                   "intensity": {"unit": "share of flights cancelled", "range": [0, 1], "doc": "announced share per strike day: a number or a list"}},
}
