from circlelog import write_mapping

with open("all_vars.pkl", "rb") as f:
    geladene_box = dill.load(f)
globals().update(geladene_box)







