import argparse
import json
import os

import numpy as np
from PIL import Image
import onnxruntime as rt
import cv2


def get_model_and_sig( model_dir ):
    with open( os.path.join( model_dir, "../signature.json" ), "r" ) as f:
        signature = json.load( f )
    model_file = "../" + signature.get( "filename" )
    if not os.path.isfile( model_file ):
        raise FileNotFoundError( f"Model file does not exist" )
    return model_file, signature


def load_model( model_file ):
    return rt.InferenceSession( path_or_bytes = model_file )


def get_prediction( image, session, signature ):
    signature_inputs = signature.get( "inputs" )
    signature_outputs = signature.get( "outputs" )
    
    if "Image" not in signature_inputs:
        raise ValueError(
            "ONNX model doesn't have 'Image' input! Check signature.json, and please report issue to Lobe." )
    
    img = process_image( image, signature_inputs.get( "Image" ).get( "shape" ) )
    
    fetches = [(key, value.get( "name" )) for key, value in signature_outputs.items()]
    # make the image a batch of 1
    feed = { signature_inputs.get( "Image" ).get( "name" ): [img[1]] }
    outputs = session.run( output_names = [name for (_, name) in fetches], input_feed = feed )
    
    results = { }
    for i, (key, _) in enumerate( fetches ):
        val = outputs[i].tolist()[0]
        if isinstance( val, bytes ):
            val = val.decode()
        results[key] = val
    
    return [results, img[0]]


def process_image( image, input_shape ):
    width, height = image.size
    
    if image.mode != "RGB":
        image = image.convert( "RGB" )
    
    left = 0
    top = 0
    right = 0
    bottom = 0
    if width != height:
        square_size = min( width, height )
        left = (width - square_size) / 2
        top = (height - square_size) / 2
        right = (width + square_size) / 2
        bottom = (height + square_size) / 2
        # Crop the center of the image
        image = image.crop( (left, top, right, bottom) )
    # now the image is square, resize it to be the right shape for the model input
    input_width, input_height = input_shape[1:3]
    if image.width != input_width or image.height != input_height:
        image = image.resize( (input_width, input_height) )
    
    image = np.asarray( image ) / 255.0
    return [[left, top, right, bottom], image.astype( np.float32 )]


def main( image, model_dir ):
    model_file, signature = get_model_and_sig( model_dir )
    session = load_model( model_file )
    prediction = get_prediction( image, session, signature )
    return prediction


# =======================================================
if __name__ == "__main__":
    # capture = cv2.VideoCapture( 'funny_dog.mp4' )
    # while True:
    #     ret, img = capture.read()
    #     if not ret:
    #         break
    #     img1 = Image.fromarray( cv2.cvtColor( img, cv2.COLOR_BGR2RGB ) )
    #     print( main( img1, os.getcwd() ) )
    #     cv2.imshow( 'mask_noMask', img )
    #     if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
    #         break
    #
    # capture.release()
    # cv2.destroyAllWindows()
    
    # ========================================
    # single image
    # =============
    # img = Image.open( "dog.4006.jpg" )
    img = cv2.imread( 'dog.4900.jpg' )
    img1 = Image.fromarray( cv2.cvtColor( img, cv2.COLOR_BGR2RGB ) )  # cv2 to pil
    result, coordinate = main( img1, os.getcwd() )
    left, top, right, bottom = coordinate[0], coordinate[1], coordinate[2], coordinate[3]
    print( 'result: {}'.format( result ) )
    print( 'coordinate: {}'.format( coordinate ) )
    img = cv2.rectangle( img, (int( left ), int( top / 4 )), (int( (left + right) / 1.2 ), int( (top + bottom) / 1.2 )),
                         (0, 255, 255), 3 )
    if result['Prediction'] == 'dog':
        cv2.putText( img, 'Dog', (int( left ), int( bottom ) - 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9,
                     (255, 255, 0), 1, cv2.LINE_AA )
    elif result['Prediction'] == 'cat':
        cv2.putText( img, 'Cat', (int( left ), int( bottom ) - 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9,
                     (255, 255, 0), 1, cv2.LINE_AA )
        # img = cv2.putText( img, 'Cat', (int( left ) + 10, int( right ) + 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9,
        #                    (255, 255, 0), 1, cv2.LINE_AA )
    else:
        print( 'unknown' )
    
    cv2.imshow( 'cat_dog', img )
    cv2.waitKey( 0 )
    cv2.destroyAllWindows()
