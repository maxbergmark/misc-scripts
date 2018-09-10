import pyaudio
import wave
import urllib.request
import time
import json

def play_sound():
	f = wave.open("service_bell.wav","rb")  
	p = pyaudio.PyAudio()
	batch = 1024
	stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
					channels = f.getnchannels(),  
					rate = f.getframerate(),  
					output = True)  
	data = f.readframes(batch)  
	while data:  
		stream.write(data)  
		data = f.readframes(batch)  
	stream.stop_stream()  
	stream.close()  
	p.terminate()

def fetch_url(n_events = [0]):
	request = urllib.request.Request('https://api2.postnord.com/rest/shipment/v1/trackandtrace/findByIdentifier.json?apikey=15845730a5462a76837692346d621009&id=UA862712419SE&locale=en')
	try:
		response = urllib.request.urlopen(request)
		json_obj = json.loads(response.read().decode("utf8"))
		events = json_obj["TrackingInformationResponse"]["shipments"][0]["items"][0]["events"]
		temp_events = len(events)
		if temp_events != n_events[0]:
			print(chr(27) + "[2J")
			print(json.dumps(events, indent=4, sort_keys=True))
			play_sound()
			n_events[0] = temp_events
	except Exception as e:
		print("something wrong", e)

while True:
	fetch_url()
	time.sleep(120)