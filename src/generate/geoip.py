from pathlib import Path

import geoip2.database
import geoip2.errors


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CITY_DB_PATH = PROJECT_ROOT / "data" / "geoip" / "GeoLite2-City.mmdb"
ASN_DB_PATH = PROJECT_ROOT / "data" / "geoip" / "GeoLite2-ASN.mmdb"


class GeoIPLookup:
    #Perform offline country and ASN lookups using MaxMind databases.

    def __init__(
        self,
        city_db_path=CITY_DB_PATH,
        asn_db_path=ASN_DB_PATH,
        ):
        # Validate that both GeoLite2 databases exist.
        self.city_db_path = Path(city_db_path)
        self.asn_db_path = Path(asn_db_path)

        if not self.city_db_path.is_file():
            raise FileNotFoundError( f"GeoLite2 City database not found: {self.city_db_path}")
        
        if not self.asn_db_path.is_file():
            raise FileNotFoundError(f"GeoLite2 ASN database not found: {self.asn_db_path}")

        # Open both databases once for repeated lookups.
        self.city_reader = geoip2.database.Reader(self.city_db_path)
        self.asn_reader = geoip2.database.Reader(self.asn_db_path)

        # Cache repeated IP lookups.
        self._cache = {}

    def lookup_country(self, ip_address):
        #Return the ISO country code for an IP address.
        try:
            response = self.city_reader.city(ip_address)
            return response.country.iso_code or "UNKNOWN"
        except (
            ValueError,
            geoip2.errors.AddressNotFoundError,
        ):
            return "UNKNOWN"

    def lookup_asn(self, ip_address):
        #Return the ASN identifier for an IP address.
        try:
            response = self.asn_reader.asn(ip_address)
            asn = response.autonomous_system_number

            if asn is None:
                return "UNKNOWN"

            return f"AS{asn}"
        except (
            ValueError,
            geoip2.errors.AddressNotFoundError,
        ):
            return "UNKNOWN"

    def lookup(self, ip_address):
        #Return cached country and ASN for an IP address.
        if ip_address in self._cache:
            return self._cache[ip_address]

        result = {
            "geo_country": self.lookup_country(ip_address),
            "asn": self.lookup_asn(ip_address),
        }

        self._cache[ip_address] = result

        return result

    def close(self):
        #Close the GeoLite2 database readers.
        self.city_reader.close()
        self.asn_reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        