from GameBoard import *

TranslateDict = {Cards.CBlack.name : "Noir",
    Cards.CBlue.name : "Bleu",
    Cards.CBrown.name : "Marron",
    Cards.CGreen.name : "Vert",
    Cards.COrange.name : "Orange",
    Cards.CPink.name : "Rose",
    Cards.CPurple.name : "Violet",
    Cards.CWhite.name : "Blanc",
    Cards.CYellow.name : "Jaune",

    ActionPawns.APSherlock.name : "Sherlock",
    ActionPawns.APAlibi.name : "Alibi",
    ActionPawns.APToby.name : "Chien Toby",
    ActionPawns.APWatson.name : "Watson",
    ActionPawns.APJoker.name : "Joker",
    ActionPawns.APReturn.name : "Tourner",
    ActionPawns.APChangeCard.name : "Echanger",
    ActionPawns.APReturn2.name : "Tourner",

    DetectivePawns.DPSherlock.name : "Sherlock",
    DetectivePawns.DPToby.name : "Chien Toby",
    DetectivePawns.DPWatson.name : "Watson",

    "up" : "Haut",
    "down" : "Bas",
    "left" : "Gauche",
    "right" : "Droite" }

def translate(string):
    if string in TranslateDict:
        return TranslateDict[string]
    else :
        return string