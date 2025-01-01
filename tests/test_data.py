from vanjari.data import Species

def test_species():
    species = Species(accession="xx", index=1, count=5)
    assert species.accession == "xx"
    assert species.index == 1
    assert species.count == 5