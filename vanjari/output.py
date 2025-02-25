from pathlib import Path
import pandas as pd
import torch
from hierarchicalsoftmax.inference import greedy_predictions, render_probabilities
from rich.progress import track
import shutil

def build_ictv_dataframe(
    probabilities_df, 
    classification_tree, 
    prediction_threshold:float=0.0, 
    image_threshold:float = 0.005, 
    output_csv:Path=None, 
    image_dir:Path=None,
    image_extension:str="png",
):
    header_string = "SequenceID,Realm (-viria),Realm_score,Subrealm (-vira),Subrealm_score,Kingdom (-virae),Kingdom_score,Subkingdom (-virites),Subkingdom_score,Phylum (-viricota),Phylum_score,Subphylum (-viricotina),Subphylum_score,Class (-viricetes),Class_score,Subclass (-viricetidae),Subclass_score,Order (-virales),Order_score,Suborder (-virineae),Suborder_score,Family (-viridae),Family_score,Subfamily (-virinae),Subfamily_score,Genus (-virus),Genus_score,Subgenus (-virus),Subgenus_score,Species (binomial),Species_score"
    header_names = header_string.split(",")

    category_names = [column for column in probabilities_df.columns if column not in ["index", "SequenceID", "original_id", "file", "chunk", "greedy_prediction"]]
    assert len(category_names) == len(classification_tree.node_list_softmax)

    classification_probabilities = torch.as_tensor(probabilities_df[category_names].to_numpy()) 

    # get greedy predictions which can use the raw activation or the softmax probabilities
    predictions = greedy_predictions(
        classification_probabilities, 
        root=classification_tree, 
        threshold=prediction_threshold,
    )

    rank_to_header = {header.split(" ")[0]:header for header in header_names[1::2]}

    data = []

    for index, node in track(enumerate(predictions), description="Building CSV output", total=len(predictions)):
        new_row = {"SequenceID": probabilities_df.loc[index, "SequenceID"]}
        
        current_probability = 1.0
        for ancestor in node.ancestors[1:] + (node,):
            if ancestor.rank == "Root":
                continue
            header = rank_to_header[ancestor.rank]

            new_row[header] = ancestor.name
            
            if ancestor.name in probabilities_df.columns:
                current_probability = probabilities_df.loc[index, ancestor.name]

            new_row[ancestor.rank+"_score"] = current_probability
        
        data.append(new_row)
    
    output_df = pd.DataFrame(data, columns=header_names).fillna("NA").replace("", "NA")

    print(output_df)

    if output_csv:
        print(f"Writing inference results to: {output_csv}")
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_csv, index=False)

    if image_dir:
        print(f"Writing inference probability renders to: {image_dir}")
        image_dir = Path(image_dir)
        if shutil.which("dot") is None:
            print("Error: Graphviz is not installed. Images will be saved as dot files instead.")
            image_extension = "dot"

        image_paths = [image_dir/f"{name}.{image_extension}" for name in probabilities_df["SequenceID"]]
        render_probabilities(
            root=classification_tree, 
            filepaths=image_paths,
            probabilities=classification_probabilities,
            predictions=predictions,
            threshold=image_threshold,
        )

    return output_df
