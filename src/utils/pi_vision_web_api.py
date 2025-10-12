"""Class wrapping PI Web API functions for data retrieval"""
import os
import requests
from requests_kerberos import HTTPKerberosAuth
import pandas as pd
from utils.data_loader import load_BI_PEM

class PIWebAPI:
    """Class wrapping PI Web API functions for data retrieval"""

    def __init__(self, base_url="https://fre-pivision/piwebapi"): # this base url runs fine locally, but it doesnt work on the server. need to find the right base url. Authentication method will likely change as well. This means the OpenShift container (Pod) cannot translate the hostname fre-pivision into an IP address. It's a fundamental DNS (Domain Name System) lookup failure.
    #def __init__(self, base_url="https://frenl01-vip009.am.boehringer.com/piwebapi"):    # this works with verify=False for now
    #def __init__(self, base_url="https://fre-pivision.eu.boehringer.com/piwebapi"):
        self.base_url = base_url
        self.security_method = HTTPKerberosAuth(
            mutual_authentication="REQUIRED", sanitize_mutual_error_response=False)
        if not os.path.exists("data"):
            os.makedirs("data")

    # Http request functions
    # ----------------------------------------------------------------
    def send_request(self, controller, action, params=None):
        """Build http request with default settings and return json response"""
        try:
            response = requests.get(
                f"{self.base_url}/{controller}/{action}",
                params=params,
                auth=self.security_method,
                verify=load_BI_PEM(),
                #verify=False,
                timeout=300,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error in request: {e}")
        return response.json()

    # PI Web API functions
    # ----------------------------------------------------------------
    def points_getbypath(self, path, selected_fields="WebId;Name;Path;Descriptor"):
        """Get point properties by its path
        generally the response would be {'WebId': 'F1DPVxiF8vuXgkWuf1S2w_rZ-wEnUAAARlJFQVMwNTAxOVxCM1BDUy5QSS0zMTcwMjkvQUkxL09VVC5DVg', 
        'Name': 'B3PCS.PI-317029/AI1/OUT.CV', 'Path': '\\\\FREAS05019\\B3PCS.PI-317029/AI1/OUT.CV', 'Descriptor': 'Feed Inlet Pressure Indicator'}"""

        # sents get request to https://fre-pivision/piwebapi/points?path=<your_path>&selectedFields=<your_fields>, path is for sensors: like this: r"\\FREAS05019\B3PCS.AIC-1390115.PROBE1"
        # The points controller is used to interact with PI Points (tags). When you want to get a point by its path, you use: GET /piwebapi/points?path=<path>
        response = self.send_request(
            "points", "", params={"path": path, "selectedFields": selected_fields}
        )

        # for selected fields, You specify the fields using a dot notation that matches the structure of the JSON response. For example:
        # WebId
        # Name
        # Path
        # Descriptor
        # Links.Self
        # Items.WebId
        # Items.Value.Timestamp
        # Items.Value.Value

        return response


    def points_attributes(self, webid, selected_fields="Items.Name;Items.Value"):
        """Get point attributes. The output of the response will contains every attribute about the sensor tag, like value, date, dataowner, tag name, parameter unit, etc"""
        response = self.send_request(
            "points",
            f"{webid}/attributes",
            params={"selectedFields": selected_fields},
        )
        return response

    # Processing Functions
    # ----------------------------------------------------------------
    def path_to_webid(self, path):
        """Get Webid from a point path (SENSOR PATH). Generally the webid would be something like F1DPVxiF8vuXgkWuf1S2w_rZ-wEnUAAARlJFQVMwNTAxOVxCM1BDUy5QSS0zMTcwMjkvQUkxL09VVC5DVg"""
        response = self.points_getbypath(path) # this is the typical workflow for getting a webid from a path      # point_info = pi_api.points_getbypath("\\PIServer\Sinusoid") # webid = point_info["WebId"]

        return response["WebId"]
    
    
    def get_recorded_data(self, webid, start_time, end_time,interval):
        """Get recorded data for a point (sensor path) with specified interval"""
        params = {
            "startTime": start_time,
            "endTime": end_time,
            "interval": interval,
            "selectedFields": "Items.Timestamp;Items.Value"
        }
        response = self.send_request("streams", f"{webid}/interpolated", params=params)
        return response



if __name__ == "__main__":
    
    unit_operations = ['affinity_chrom','AEX','CEX','TFF']

    user_select_unit_operation = "TFF"

    # select unit op to get the data
    if user_select_unit_operation == "TFF":
        tag_file = pd.read_excel("tff_pi_tags.xlsx", sheet_name="Sheet1", engine="openpyxl")

        tag_paths = tag_file["tag_name"].tolist()

        # user input for multiple time ranges
        time_ranges = [["9/27/2025 22:46:00", "9/27/2025 23:29:00"], ["9/17/2025 13:29:00", "9/17/2025 14:35:00"]]

        # Initialize API
        pi_api = PIWebAPI()

        # Collect data
        all_data = []

        for time_range in time_ranges:
            for path in tag_paths:
                try:
                    point_info = pi_api.points_getbypath(path)
                    webid = point_info["WebId"]
                    tag_name = point_info["Name"]
                    recorded_data = pi_api.get_recorded_data(webid, time_range[0], time_range[1],"60s")
                    for item in recorded_data.get("Items", []):
                        all_data.append({
                                        "tag": tag_name,
                                        "timestamp": item["Timestamp"],
                                        "value": item["Value"]
                                    })
                except Exception as e:
                    print(f"Failed to retrieve data for {path}: {e}")

        # Save to CSV
        df = pd.DataFrame(all_data)


        # map parameter name to tag name
        tag_name_no_prefix = tag_file["tag_name"].str.split("\\").str[-1]
        tag_to_param = dict(zip(tag_name_no_prefix, tag_file["tag_parameter_name"]))
        df['parameter_name'] = df['tag'].map(tag_to_param)

        # convert timestamp to datetime for pacific time
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp_local"] = df["timestamp"].dt.tz_convert("America/Los_Angeles").dt.strftime("%Y-%m-%d %H:%M:%S")

        # save to csv
        df.to_csv("data/pi_data.csv")

        print("Data retrieval complete. Saved to data/pi_data.csv.")