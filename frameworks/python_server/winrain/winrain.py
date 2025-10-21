#encoding:UTF-8
from gunicorn.http.body import ChunkedReader, LengthReader, EOFReader, Body
from functools import wraps
import re
import traceback
from . import winlogger
import urllib.parse

HEADER_FILED = ['QUERY_STRING', 'HTTP_HOST', 'REQUEST_METHOD', 'RAW_URI', 'CONTENT_TYPE', 'HTTP_LOG_ID']

PROCESS_DEBUG = 0
class WinRequest(object):
    def __init__(self, env=None):
        self.header_dict = {}
        self.query_param_dict = {}
        if env is not None:
            self._parse_request(env)

    def _parse_request(self, env):
        self._parse_header(env)
        self._parse_url_param(env)

    def _parse_header(self, env):
        for field in HEADER_FILED:
            if field in env:
                self.header_dict[field] = env[field]
        if PROCESS_DEBUG:
            winlogger.debug('header_dict:' + str(self.header_dict))

    def _parse_url_param(self, env):
        if 'QUERY_STRING' not in env:
            return
        query_string = env['QUERY_STRING']
        #query_string = urllib.parse.unquote(query_string)
        if PROCESS_DEBUG:
            winlogger.debug('query_string:' + str(query_string))
        field_list = query_string.split('&')
        for field in field_list:
            itr = field.split('=')
            if len(itr) == 2:
                key = urllib.parse.unquote(itr[0])
                value = urllib.parse.unquote(itr[1]) 
                #self.query_param_dict[itr[0]] = itr[1]
                self.query_param_dict[key] = value
            elif len(itr) == 1:
                key = urllib.parse.unquote(itr[0])
                #self.query_param_dict[itr[0]] = ''
                self.query_param_dict[key] = ''
        if PROCESS_DEBUG:
            winlogger.debug('query_param_dict:' + str(self.query_param_dict))

    def get_url_param(self, key):
        if key in self.query_param_dict:
            return self.query_param_dict(key)

    def get_url_params(self):
        return self.query_param_dict
    
    def get_header_field(self, key):
        if key in self.header_dict:
            return self.header_dict[key]

class WinRain(object):
    def __init__(self):
        self.functions = {}
        self.request = WinRequest()
        self.http_body = ''
        pass

    def add_uri(self, uri):
        def decorator(func):
            self.functions[uri] = func
        return decorator

    def _map_func(self, uri):
        if uri in self.functions:
            return self.functions[uri]
        else:
            for key in self.functions:
                if re.match(key, uri):
                    self.functions[uri] = self.functions[key]
                    return self.functions[key]

    def get_body(self):
        return self.http_body
        
    def process(self, env, start_response):
        if PROCESS_DEBUG:
            winlogger.debug('env:' + str(env))
        self.request = WinRequest(env)
        body = env["wsgi.input"]
        self.http_body = body.read()
        uri = env['PATH_INFO']
        log_id = self.request.get_header_field('HTTP_LOG_ID')
        if log_id != None:
            winlogger._set_log_id(log_id)
        func = self._map_func(uri)
        if func is None:
            winlogger.error("not exist the uri[%s]" % uri)
            data = 'Hello World!\n'
        else:
            try:
                data = func()
            except:
                winlogger.error(func.__name__ + " process exception")
                winlogger.error(traceback.format_exc())
                data = "500 Internal Server Error"
                start_response("500 Internal Server Error", [
                        ("Content-Type", "text/html; charset=UTF-8"),
                        ("Content-Length", str(len(data)))
                    ])
                return iter([data.encode('utf-8')])
        start_response("200 OK", [
            ("Content-Type", "text/html; charset=UTF-8")
            ])
#        return iter([bytes(data, 'ascii')])
        return iter([data.encode('utf-8')])

    def __call__(self, env, start_response):
        return self.process(env, start_response)

