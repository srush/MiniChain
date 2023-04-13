# + tags=["hide_inp"]
desc = """
### Table

"""
# -

# $
import pandas as pd
from minichain import prompt, Mock, show, OpenAIStream
import minichain
import json
import gradio as gr

rotowire = json.load(open("data.json"))
names = {
    '3-pointer percentage': 'FG3_PCT',
    '3-pointers attempted': 'FG3A',
    '3-pointers made': 'FG3M',
    'Assists': 'AST',
    'Blocks': 'BLK',
    'Field goal percentage': 'FG_PCT',
    'Field goals attempted': 'FGA',
    'Field goals made': 'FGM',
    'Free throw percentage': 'FT_PCT',
    'Free throws attempted': 'FTA',
    'Free throws made': 'FTM',
    'Minutes played': 'MIN',
    'Personal fouls': 'PF',
    'Points': 'PTS',
    'Rebounds': 'REB',
    'Rebounds (Defensive)': 'DREB',
    'Rebounds (Offensive)': 'OREB',
    'Steals': 'STL',
    'Turnovers': 'TO'
}
# Convert an example to dataframe
def to_df(d):
    players = {player for v in d.values() if v is not None for player, _  in v.items()}
    lookup = {k: {a: b for a, b in v.items()} for k,v in d.items()}
    rows = [{"player": p} | {k: "_" if p not in lookup.get(k, []) else lookup[k][p] for k in names.keys()}
            for p in players]
    return pd.DataFrame.from_dict(rows).astype("str").sort_values(axis=0, by="player", ignore_index=True).transpose()


# Make few shot examples
few_shot_examples = 2
examples = []
for i in range(few_shot_examples):
    examples.append({"input": rotowire[i][1],
                     "output": to_df(rotowire[i][0][1]).transpose().set_index("player").to_csv(sep="\t")})

@prompt(OpenAIStream(),
        template_file="table.pmpt.txt",
        block_output=gr.HTML,
        stream=True)
def extract(model, passage, typ):
    state = []
    out = ""
    for token in model.stream(dict(player_keys=names.items(), examples=examples, passage=passage, type=typ)):
        out += token
        html = "<table><tr><td>" + out.replace("\t", "</td><td>").replace("\n", "</td></tr><tr><td>")  + "</td></td></table>"
        yield html
    yield html



def run(query):
    return extract(query, "Player")

# $

gradio = show(run,
              examples = [rotowire[i][1] for i in range(50, 55)],
              subprompts=[extract],
              code=open("table.py", "r").read().split("$")[1].strip().strip("#").strip(),
              out_type="markdown"
            )

if __name__ == "__main__":
    gradio.queue().launch()
