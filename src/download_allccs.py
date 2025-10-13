import requests
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the Data")
    parser.add_argument("--user", type=str, default="", help="user_name")
    parser.add_argument("--passw", type=str, default="", help="passwords")
    parser.add_argument(
        "--output",
        type=str,
        default="./data/origin/allccs_download.csv",
        help="path to output data",
    )
    args = parser.parse_args()

    s = requests.Session()
    login_url = "http://allccs.zhulab.cn/login"
    from_data = {"User": args.user, "Pass": args.passw}

    # Login
    login_r = s.post(login_url, data=from_data)
    print("Login Response: ", login_r)

    # Download
    # AllCCS00000000 - AllCCS02049881
    for req_i in range(2049881 // 100 + 1):
        url = "http://allccs.zhulab.cn/database/browser_download?ids="
        ids = [
            "AllCCS" + str(i).zfill(8) for i in range(req_i * 100, req_i * 100 + 100)
        ]
        url += ",".join(ids)
        print(">>>", req_i * 100, req_i * 100 + 100)

        r = s.get(url)
        print("Get response: ", r)

        with open(args.output, "ab+") as f:
            f.write(r.content)
