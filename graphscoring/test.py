#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys
import numpy as np
from nltk.tokenize import sent_tokenize

import csv


def get_sentences(anytext):
    sent_list = sent_tokenize(anytext)

    export(sent_list)


def export(sent_list):
    with open("abc.csv", "a") as output:
        writer = csv.writer(output, lineterminator='\n')
        for sent in sent_list:
            writer.writerow([sent])

# u = "It’s going to be business as usual for Chief Justice of India Dipak Misra in the week that starts April 23, according to the list released by the Supreme Court’s registry on Saturday. On Friday, the Congress and six other parties submitted notice of a motion for the removal of the CJI to chairman of the Rajya Sabha M Venkaiah Naidu. This is the first such notice ever presented against a CJI. The notice was signed by 71 MPs, although seven of them have retired from he upper house since signing it. Naidu is yet to take a call on admitting the motion. Government officials said on Saturday that the Congress had flouted Rajya Sabha rules by going public with the notice of a motion that was yet to be admitted in the house. It’s been six months since their wedding but Virat Kohli is still swooning over wife Anushka Sharma. Of course, we are not surprised. The country’s most adored couple often shares pictures with each other on social media and the latest example is just too cute. The cricketer posted a paparazzi picture of himself with his actor wife on Friday. “Such a stunner, Love of my life! @anushkasharma,” he captioned the photo with ‘heart-eyes’ emoji. The couple looks like a monochromatic dream with Anushka seen in a black dress and Virat in a crisp white shirt. Manchester United came from behind to beat Tottenham Hotspur 2-1 at Wembley on Saturday and secure their place in the FA Cup final. Ander Herrera struck the winner midway through the second half, after Alexis Sanchez had cancelled out an opening goal from Dele Alli. United lost on their league visit to Spurs this season after falling behind to an early goal and it looked like Mauricio Pochettino’s side could repeat the feat when Alli turned home Christian Eriksen’s cross. Prime Minister Narendra Modi Saturday honoured bureaucrats for making Manipur’s Karang the first cashless island of the country, and implementing the Goods and Services Tax (GST) and other priority initiatives of the Centre. Karang island, a remote and backward region, was cut-off from the (Bishnupur) district due to insurgency for a long time. Incentives were provided for training towards digital payments and five POS machines were installed on the island, the award citation read. An online channel was launched to make people aware about digital payments. In addition, social media interventions were made for the purpose, according to the citation. As a result, 92% of the bank accounts were seeded with mobile and 70% of them were seeded with Aadhaar. The percentage of the electricity bills paid digitally increased from 78% to 97% in the last 20 months, it said. Modi gave the award to Deputy Commissioner of the district for promoting digital payment. Shiv Sena president Uddhav Thackeray on Saturday said he was not a detractor of Prime Minister Narendra Modi, but will always speak up when he doesn’t approve of something. I am not a critic of Modi, but I will speak on the issues where I don’t agree with Modi government’s decisions, he said. Sena, an ally of the BJP at the Centre and in Maharashtra, continually takes swipe at the prime minister and his party, especially through its mouthpiece `Saamana’. Speaking at the release of a Marathi book, ‘Gof’, penned by Sena’s Rajya Sabha MP and `Saamana’ executive editor Sanjay Raut, Thackeray said his father had taught him to speak his mind. An Indian-American teenager, wanted on a felony firearms possession warrant, was shot dead by a California police after he opened fire on them, officials said.The individual, identified as Nathaniel Prasad, 18, was shot dead on April 5, according to an investigation report released by the Freemont Police Department on Thursday. He was wanted on a felony probation warrant and a misdemeanour evading arrest warrant for fleeing from a Fremont School Resource Officer on March 22. On April 5, the Fremont Police Department’s Street Crimes Unit identified Prasad as a passenger in a vehicle that was being driven by a female in Fremont area. The vehicle was known to be associated with Prasad. Soon the information was broadcast over the police radio about the vehicle and the active warrants and requested marked patrol units to assist with a traffic stop."
# u = u.encode('ascii', 'ignore').decode('ascii')
# get_sentences(u)

#
#
# input = ['a', 'b', 'c', 'd']
# res = [0, 1, 0, 1]
#
# input.insert(0, "Text")
# res.insert(0, "Prediction")
#
# rows = zip(input, res)
#
# with open("test_predict.csv", 'w') as f:
#     writer = csv.writer(f)
#     for row in rows:
#         writer.writerow(row)




