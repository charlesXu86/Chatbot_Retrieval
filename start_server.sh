#!/usr/bin/env bash

nohup python manage.py runserver 192.168.0.130:9005 > /home/log/faq.log 2>&1 &

# nohup python manage.py runserver 172.18.86.20:9007 > /dev/null