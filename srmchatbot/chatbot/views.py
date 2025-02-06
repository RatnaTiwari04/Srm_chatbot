import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

def index(request):
    return render(request, 'chatbot.html')

@csrf_exempt
def get_response(request):
    if request.method == 'POST':
        user_message = json.loads(request.body)['message']
        with open('chatbot_data.json', 'r') as f:
            responses = json.load(f)
        
        response = responses.get(user_message.lower(), responses['default'])
        return JsonResponse({'message': response})