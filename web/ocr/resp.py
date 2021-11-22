import json
from django.http import HttpResponse


def get_json_http_resp(data, status, info):
    resp = {"data": data, "status": status, "info": info}
    return HttpResponse(json.dumps(resp), content_type='application/json')


def get_suc_http_resp(data=None, info='succeed.'):
    return get_json_http_resp(data, 200, info)


def get_fail_http_resp(data=None, info="failed."):
    return get_json_http_resp(data, -1, info)
