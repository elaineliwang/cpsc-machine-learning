#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 22:22:18 2018

@author: Elaine
"""
# functions to clean the data

def change_year(year):
    if year == '4+':
        return 4
    else:
        return int(year)

def change_gender(gender):
    if gender == 'F':
        return 0
    else:
        return 1
def change_age(age):
    if age == '0-17':
        return 0
    elif age == '18-25':
        return 1
    elif age == '26-35':
        return 2
    elif age == '36-45':
        return 3
    elif age == '46-50':
        return 4
    elif age == '51-55':
        return 5
    else:
        return 6
def change_city(city_category):
    if city_category == 'A':
        return 0
    elif city_category == 'B':
        return 1
    else:
        return 2
