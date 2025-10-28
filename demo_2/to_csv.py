import re
import pandas as pd
from pathlib import Path

LOG_PATH = "log/log_2025-10-22_13-48-21.txt"


def parse_log(log_text):
    results = []
    blocks = re.split(r"\n🧩 Config:", log_text)
    for block in blocks[1:]:
        config_match = re.search(r"(\w+).*window=(\d+), stride=(\d+)", block)
        if not config_match:
            continue
        config_name = config_match.group(1)
        window = int(config_match.group(2))
        stride = int(config_match.group(3))

        channels = re.search(r"Selected Channels:\s*(\[.*?\])", block)
        data_proc = re.search(r"Data Processing:\s*(.*)", block)
        selected_indices = channels.group(1) if channels else "?"
        data_processing = data_proc.group(1).strip() if data_proc else "None"

        def extract_section(title):
            pattern = rf"📊 {title} Results[\s\S]*?(?=\n📊|\Z)"
            section = re.search(pattern, block)
            result = {}
            if not section:
                return result
            text = section.group(0)
            overall = re.search(r"Overall Accuracy:\s*([\d.]+)%", text)
            if overall:
                result["Overall"] = float(overall.group(1))
            for line in text.splitlines():
                m = re.match(r"(\w+)\s*:\s*([\d.]+)%", line.strip())
                if m:
                    result[m.group(1)] = float(m.group(2))
            return result

        train_acc = extract_section("Train")
        test_acc = extract_section("Test")

        online_blocks = re.findall(r"📄 File:[\s\S]*?🏁 Final Decision:.*", block)
        online_details = []
        vote_scores = []

        for online_block in online_blocks:
            f_match = re.search(r"File:\s*(\S+)", online_block)
            truth_match = re.search(r"True label:\s*(\w+)", online_block)
            final_match = re.search(r"Final Decision:\s*(.*)", online_block)
            pred_label = (
                re.match(r"(\w+)", final_match.group(1)).group(1)
                if final_match
                else "?"
            )
            correct = (
                "✅" if truth_match and truth_match.group(1) == pred_label else "❌"
            )
            frac_match = re.search(r"\((\d+)\s*/\s*(\d+)\)", final_match.group(1))
            frac = (
                float(frac_match.group(1)) / float(frac_match.group(2))
                if frac_match
                else 0
            )
            vote_score = frac if correct == "✅" else -frac
            vote_scores.append(vote_score)

            votes = re.findall(r"(\w+)\s*:\s*(\d+)\s*windows", online_block)
            vote_str = ", ".join([f"{cls}:{cnt}" for cls, cnt in votes])

            detail = (
                f"[{f_match.group(1) if f_match else '?'}] "
                f"{vote_str} → {pred_label} "
                f"({frac_match.group(1) if frac_match else '?'} / {frac_match.group(2) if frac_match else '?'}) {correct}"
            )
            online_details.append(detail)

        final_vote_acc = (
            round(sum(vote_scores) / len(vote_scores), 3) if vote_scores else None
        )
        train_overall = train_acc.get("Overall", 0.0)
        test_overall = test_acc.get("Overall", 0.0)
        overfit_gap = round(train_overall - test_overall, 2)

        row = {
            "Config": config_name,
            "Window": window,
            "Stride": stride,
            "Channels": selected_indices,
            "Data_Processing": data_processing,
            **{f"Train_{k}": v for k, v in train_acc.items()},
            **{f"Test_{k}": v for k, v in test_acc.items()},
            "Online_Details": " | ".join(online_details),
            "Final_Vote_Accuracy": final_vote_acc,
            "Overfitting_Gap": overfit_gap,
        }
        results.append(row)

    # df = pd.DataFrame(results)
    # df = df.sort_values(by="Final_Vote_Accuracy", ascending=False, na_position="last")
    # return df
    df = pd.DataFrame(results)

    df["Overfitting_Gap"] = df["Overfitting_Gap"].fillna(999)
    df["Final_Vote_Accuracy"] = df["Final_Vote_Accuracy"].fillna(-999)

    df["Abs_Overfitting_Gap"] = df["Overfitting_Gap"].abs()

    df = df.sort_values(
        by=["Final_Vote_Accuracy", "Abs_Overfitting_Gap"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":
    text = Path(LOG_PATH).read_text(encoding="utf-8")
    df = parse_log(text)

    out_path = LOG_PATH.replace(".txt", "_final_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Saved final summary to {out_path}")
    print(df[["Config", "Data_Processing", "Final_Vote_Accuracy", "Overfitting_Gap"]])
