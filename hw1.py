#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:22:45 2017

@author: jingkong
"""

###E6998 – HW#1
##Due 9/26/2017 at 11:59PM (23:59) - EST

##hw1

##Ex1
##part a
####method 1
sent = ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']
s = [i for i in sent if i[0:2]=="sh"]
s
##part b
m = [i for i in sent if len(i) > 4]
m


##Ex2
####method 1
text = "Wild brown trout are elusive"
li = list()
l = text.split(" ")##this is actually done
for i in l:
   li.append(i)
li

###method 2
li = list()
li.append("Wild")
li.append("brown")
li.append("trout")
li.append("are")
li.append("elusive")
li







