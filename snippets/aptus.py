from viewstate import ViewState
import urllib.request

vs1 = "/wEPDwUKLTY4MjI1MDA4NA9kFgICAw9kFgICAQ9kFgJmDxYCHgtfIUl0ZW1Db3VudAICFgRmD2QWAgIBDw8WBB4ISW1hZ2VVcmwFIH4vaW1hZ2VzL0xhbmd1YWdlSWNvbnMvc3Ytc2UuZ2lmHg9Db21tYW5kQXJndW1lbnQFBXN2LVNFZGQCAQ9kFgICAQ8PFgQfAQUgfi9pbWFnZXMvTGFuZ3VhZ2VJY29ucy9lbi1nYi5naWYfAgUFZW4tR0JkZBgBBR5fX0NvbnRyb2xzUmVxdWlyZVBvc3RCYWNrS2V5X18WAwUwY3RsMDIkbGFuZ3VhZ2VSZXBlYXRlciRjdGwwMCRpbWFnZUJ1dHRvbkxhbmd1YWdlBTBjdGwwMiRsYW5ndWFnZVJlcGVhdGVyJGN0bDAxJGltYWdlQnV0dG9uTGFuZ3VhZ2UFHExvZ2luUG9ydGFsJExvZ2luSW1hZ2VCdXR0b27mxoduUjO73E1fxorgw8lFyTq/QQ=="
vs2 = "/wEPDwUKLTY4MjI1MDA4NA9kFgICAw9kFgICAQ9kFgJmDxYCHgtfIUl0ZW1Db3VudAICFgRmD2QWAgIBDw8WBB4ISW1hZ2VVcmwFIH4vaW1hZ2VzL0xhbmd1YWdlSWNvbnMvc3Ytc2UuZ2lmHg9Db21tYW5kQXJndW1lbnQFBXN2LVNFZGQCAQ9kFgICAQ8PFgQfAQUgfi9pbWFnZXMvTGFuZ3VhZ2VJY29ucy9lbi1nYi5naWYfAgUFZW4tR0JkZBgBBR5fX0NvbnRyb2xzUmVxdWlyZVBvc3RCYWNrS2V5X18WAwUwY3RsMDIkbGFuZ3VhZ2VSZXBlYXRlciRjdGwwMCRpbWFnZUJ1dHRvbkxhbmd1YWdlBTBjdGwwMiRsYW5ndWFnZVJlcGVhdGVyJGN0bDAxJGltYWdlQnV0dG9uTGFuZ3VhZ2UFHExvZ2luUG9ydGFsJExvZ2luSW1hZ2VCdXR0b27mxoduUjO73E1fxorgw8lFyTq/QQ=="
vs3 = "/wEPDwUKLTY4MjI1MDA4NA9kFgICAw9kFgICAQ9kFgJmDxYCHgtfIUl0ZW1Db3VudAICFgRmD2QWAgIBDw8WBB4ISW1hZ2VVcmwFIH4vaW1hZ2VzL0xhbmd1YWdlSWNvbnMvc3Ytc2UuZ2lmHg9Db21tYW5kQXJndW1lbnQFBXN2LVNFZGQCAQ9kFgICAQ8PFgQfAQUgfi9pbWFnZXMvTGFuZ3VhZ2VJY29ucy9lbi1nYi5naWYfAgUFZW4tR0JkZBgBBR5fX0NvbnRyb2xzUmVxdWlyZVBvc3RCYWNrS2V5X18WAwUwY3RsMDIkbGFuZ3VhZ2VSZXBlYXRlciRjdGwwMCRpbWFnZUJ1dHRvbkxhbmd1YWdlBTBjdGwwMiRsYW5ndWFnZVJlcGVhdGVyJGN0bDAxJGltYWdlQnV0dG9uTGFuZ3VhZ2UFHExvZ2luUG9ydGFsJExvZ2luSW1hZ2VCdXR0b27mxoduUjO73E1fxorgw8lFyTq/QQ=="
ev3 = "/wEWBgLUkfX7CgKBx8WVDwKk3rbwDAKm6uySCAKlnPOvBAKmw8iKAse1npqQxeNtE0tIgHnnO30AYpUx"
ev4 = "/wEWBgLUkfX7CgKBx8WVDwKk3rbwDAKm6uySCAKlnPOvBAKmw8iKAse1npqQxeNtE0tIgHnnO30AYpUx"

print(ViewState(vs1).decode())
print(ViewState(vs2).decode())
print(vs1 == vs3)

cookieProcessor = urllib.request.HTTPCookieProcessor()
opener = urllib.request.build_opener(cookieProcessor)

request = urllib.request.Request('http://aptusportal.sssb.se/')
response = opener.open(request,timeout=100)

data = {
	"__LASTFOCUS": "",
	"__EVENTTARGET": "",
	"__EVENTARGUMENT": "",
	"__VIEWSTATE": vs3,
	"__EVENTVALIDATION": ev3,
	"LoginPortal$UserName": "1202-0905-001",
	"LoginPortal$Password": "1202-0905-001",
	"LoginPortal$LoginButton": "Logga in"
}

request = urllib.request.Request(r'http://aptusportal.sssb.se/login.aspx?ReturnUrl=%2findex.aspx', headers = data)
response = opener.open(request,timeout=100)

# response = urllib.request.urlopen(request)
json_obj = response.read().decode('utf8')
print(json_obj)
