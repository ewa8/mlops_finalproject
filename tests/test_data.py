import src.data.make_dataset as md

def test_divideFilename():

    filename = 'yes 56.jpg'
    
    f, e = md.findFileExtension(filename)

    assert f =='yes 56'
    assert e =='jpg'
    

