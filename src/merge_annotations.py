import os
import pandas as pd
import argparse


root = "/work3/s203257"
paths = {
    "Nepal": "/work1/fbohy/_Data for Frederik and Chris/Nepal video data - Frames and annotation/annotations/",
    "Myanmar": "",
}


def merged_file(path: str):
    df = pd.DataFrame()
    for file in os.listdir(path):
        if file.endswith(".csv"):
            csv = pd.read_csv(os.path.join(path, file))
            csv["video_id"] = file.split(".")[0]
            df = df.append(csv)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "Project",
        metavar="project",
        type=str,
    )
    args = parser.parse_args()

    project = args.Project

    if project == "Nepal":
        df = merged_file(paths["Nepal"])

        columnsTitles = [
            "video_id",
            "tracker_id",
            "frame_id",
            "x",
            "y",
            "w",
            "h",
            "label",
        ]
        df = df.reindex(columns=columnsTitles)
        df = df.rename(columns={"tracker_id": "track_id"})

        df.to_csv(f"{root}/Nepal_annotation_merged.csv", index=False)

    if project == "Myanmar":
        df = merged_file(paths["Myanmar"])

        columnsTitles = [
            "video_id",
            "track_id",
            "frame_id",
            "x",
            "y",
            "w",
            "h",
            "label",
        ]
        df = df.reindex(columns=columnsTitles)

        df.to_csv(f"{root}/Myanmar_annotation_merged.csv", index=False)
