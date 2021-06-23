import src.data.make_dataset as md

def test_divideFilename():

    filename = 'yes56.jpg'
    
    f, e = md.findFileExtension(filename)

    assert f =='yes56'
    assert e =='jpg'
    

