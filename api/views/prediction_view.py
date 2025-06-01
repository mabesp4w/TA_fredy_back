from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from api.utils.predict import predict_single_audio

class PredictionViewSet(APIView):
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request, *args, **kwargs):
        try:
            # Check if file is in request
            if 'audio_file' not in request.FILES:
                return Response(
                    {'error': 'No audio file provided'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            audio_file = request.FILES['audio_file']
            
            # Call predict_single_audio function
            prediction_result = predict_single_audio(audio_file)
            
            if prediction_result is None:
                return Response(
                    {'error': 'Failed to process audio file'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Return prediction results
            return Response(prediction_result, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )