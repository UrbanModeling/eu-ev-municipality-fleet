# -*- coding: utf-8 -*-
"""
NUTS-2 level validation of the XGBoost downscaling model.
Compares aggregated model predictions against Eurostat total car stock
(dataset road_eqs_carage, CAR, NR) for 7 countries with > 10 NUTS-2 regions.

Countries: Germany (38), France (22), Italy (21), Spain (19),
           United Kingdom (36), Netherlands (12), Belgium (11)

Uses pre-computed predictions from output/eu14_city_vehicles_xgb_2011_2023.csv.
Output: output/validation_nuts2/
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd, numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROJ = Path(__file__).parent.parent

# ── USER CONFIGURATION ────────────────────────────────────────────────────
# Set these paths to the locations of your downloaded Eurostat validation data.
# See README for data sources.
ESTAT_TSV  = Path("path/to/eurostat_nuts3_vehicles.tsv")
LABELS_TSV = Path("path/to/eurostat_nuts_labels.tsv")
# ──────────────────────────────────────────────────────────────────────────

VEH_CSV = PROJ / 'output' / 'eu14_city_vehicles_xgb_2011_2023.csv'
BND_CSV = PROJ / 'data' / 'boundary' / 'eu14_city.csv'
OUT_DIR = PROJ / 'output' / 'validation_nuts2'
OUT_DIR.mkdir(exist_ok=True)

# ── NUTS-2 mapping tables ──────────────────────────────────────────────────

# Germany: NAME_1 (single-NUTS2 states) and NAME_2 (districts in multi-NUTS2 states)
DE_NAME1 = {
    'Berlin': 'DE30', 'Brandenburg': 'DE40', 'Bremen': 'DE50', 'Hamburg': 'DE60',
    'Mecklenburg-Vorpommern': 'DE80', 'Saarland': 'DEC0',
    'Sachsen-Anhalt': 'DEE0', 'Schleswig-Holstein': 'DEF0', 'Thüringen': 'DEG0',
}
DE_NAME2 = {
    'Stuttgart':'DE11','Böblingen':'DE11','Esslingen':'DE11','Göppingen':'DE11',
    'Ludwigsburg':'DE11','Rems-Murr-Kreis':'DE11','Heilbronn':'DE11',
    'Heilbronn (Stadtkreis)':'DE11','Hohenlohekreis':'DE11','Schwäbisch Hall':'DE11',
    'Main-Tauber-Kreis':'DE11','Heidenheim':'DE11','Ostalbkreis':'DE11',
    'Baden-Baden':'DE12','Karlsruhe':'DE12','Karlsruhe (Stadtkreis)':'DE12',
    'Rastatt':'DE12','Heidelberg':'DE12','Mannheim':'DE12',
    'Neckar-Odenwald-Kreis':'DE12','Rhein-Neckar-Kreis':'DE12','Pforzheim':'DE12',
    'Calw':'DE12','Enzkreis':'DE12','Freudenstadt':'DE12',
    'Freiburg im Breisgau':'DE13','Breisgau-Hochschwarzwald':'DE13',
    'Emmendingen':'DE13','Ortenaukreis':'DE13','Rottweil':'DE13',
    'Schwarzwald-Baar-Kreis':'DE13','Tuttlingen':'DE13','Konstanz':'DE13',
    'Lörrach':'DE13','Waldshut':'DE13',
    'Reutlingen':'DE14','Tübingen':'DE14','Zollernalbkreis':'DE14','Ulm':'DE14',
    'Alb-Donau-Kreis':'DE14','Biberach':'DE14','Bodenseekreis':'DE14',
    'Bodensee':'DE14','Ravensburg':'DE14','Sigmaringen':'DE14',
    'München':'DE21','München (Kreisfreie Stadt)':'DE21','Altötting':'DE21',
    'Berchtesgadener Land':'DE21','Bad Tölz-Wolfratshausen':'DE21','Dachau':'DE21',
    'Ebersberg':'DE21','Eichstätt':'DE21','Erding':'DE21','Freising':'DE21',
    'Fürstenfeldbruck':'DE21','Garmisch-Partenkirchen':'DE21','Ingolstadt':'DE21',
    'Landsberg am Lech':'DE21','Miesbach':'DE21','Mühldorf am Inn':'DE21',
    'Neuburg-Schrobenhausen':'DE21','Pfaffenhofen an der Ilm':'DE21',
    'Rosenheim':'DE21','Rosenheim (Kreisfreie Stadt)':'DE21','Starnberg':'DE21',
    'Traunstein':'DE21','Weilheim-Schongau':'DE21',
    'Deggendorf':'DE22','Dingolfing-Landau':'DE22','Freyung-Grafenau':'DE22',
    'Kelheim':'DE22','Landshut':'DE22','Landshut (Kreisfreie Stadt)':'DE22',
    'Passau':'DE22','Passau (Kreisfreie Stadt)':'DE22','Regen':'DE22',
    'Rottal-Inn':'DE22','Straubing':'DE22','Straubing-Bogen':'DE22',
    'Amberg':'DE23','Amberg-Sulzbach':'DE23','Cham':'DE23',
    'Neumarkt in der Oberpfalz':'DE23','Neustadt an der Waldnaab':'DE23',
    'Regensburg':'DE23','Regensburg (Kreisfreie Stadt)':'DE23','Schwandorf':'DE23',
    'Tirschenreuth':'DE23','Weiden in der Oberpfalz':'DE23',
    'Bamberg':'DE24','Bamberg (Kreisfreie Stadt)':'DE24','Bayreuth':'DE24',
    'Bayreuth (Kreisfreie Stadt)':'DE24','Coburg':'DE24','Coburg (Kreisfreie Stadt)':'DE24',
    'Forchheim':'DE24','Hof':'DE24','Hof (Kreisfreie Stadt)':'DE24',
    'Kronach':'DE24','Kulmbach':'DE24','Lichtenfels':'DE24',
    'Wunsiedel im Fichtelgebirge':'DE24',
    'Ansbach':'DE25','Ansbach (Kreisfreie Stadt)':'DE25','Erlangen':'DE25',
    'Erlangen-Höchstadt':'DE25','Fürth':'DE25','Fürth (Kreisfreie Stadt)':'DE25',
    'Nürnberg':'DE25','Nürnberger Land':'DE25','Roth':'DE25','Schwabach':'DE25',
    'Weißenburg-Gunzenhausen':'DE25','Neustadt an der Aisch-Bad Windsheim':'DE25',
    'Aschaffenburg':'DE26','Aschaffenburg (Kreisfreie Stadt)':'DE26',
    'Bad Kissingen':'DE26','Haßberge':'DE26','Kitzingen':'DE26','Miltenberg':'DE26',
    'Main-Spessart':'DE26','Rhön-Grabfeld':'DE26','Schweinfurt':'DE26',
    'Schweinfurt (Kreisfreie Stadt)':'DE26','Würzburg':'DE26',
    'Würzburg (Kreisfreie Stadt)':'DE26',
    'Augsburg':'DE27','Augsburg (Kreisfreie Stadt)':'DE27','Aichach-Friedberg':'DE27',
    'Dillingen an der Donau':'DE27','Donau-Ries':'DE27','Günzburg':'DE27',
    'Kaufbeuren':'DE27','Kempten (Allgäu)':'DE27','Lindau (Bodensee)':'DE27',
    'Memmingen':'DE27','Neu-Ulm':'DE27','Oberallgäu':'DE27',
    'Ostallgäu':'DE27','Unterallgäu':'DE27',
    'Darmstadt':'DE71','Darmstadt-Dieburg':'DE71','Frankfurt am Main':'DE71',
    'Groß-Gerau':'DE71','Hochtaunuskreis':'DE71','Main-Kinzig-Kreis':'DE71',
    'Main-Taunus-Kreis':'DE71','Odenwaldkreis':'DE71','Offenbach':'DE71',
    'Offenbach am Main':'DE71','Rheingau-Taunus-Kreis':'DE71',
    'Wetteraukreis':'DE71','Bergstraße':'DE71','Wiesbaden':'DE71',
    'Gießen':'DE72','Lahn-Dill-Kreis':'DE72','Limburg-Weilburg':'DE72',
    'Marburg-Biedenkopf':'DE72','Vogelsbergkreis':'DE72',
    'Fulda':'DE73','Hersfeld-Rotenburg':'DE73','Kassel':'DE73',
    'Kassel (Kreisfreie Stadt)':'DE73','Schwalm-Eder-Kreis':'DE73',
    'Waldeck-Frankenberg':'DE73','Werra-Meißner-Kreis':'DE73',
    'Braunschweig':'DE91','Gifhorn':'DE91','Goslar':'DE91','Helmstedt':'DE91',
    'Peine':'DE91','Salzgitter':'DE91','Wolfenbüttel':'DE91','Wolfsburg':'DE91',
    'Region Hannover':'DE92','Diepholz':'DE92','Hameln-Pyrmont':'DE92',
    'Hildesheim':'DE92','Holzminden':'DE92','Nienburg (Weser)':'DE92',
    'Northeim':'DE92','Schaumburg':'DE92','Osterode am Harz':'DE92',
    'Celle':'DE93','Cuxhaven':'DE93','Harburg':'DE93','Heidekreis':'DE93',
    'Lüchow-Dannenberg':'DE93','Lüneburg':'DE93','Osterholz':'DE93',
    'Rotenburg (Wümme)':'DE93','Stade':'DE93','Uelzen':'DE93','Verden':'DE93',
    'Ammerland':'DE94','Aurich':'DE94','Cloppenburg':'DE94','Delmenhorst':'DE94',
    'Emsland':'DE94','Emden':'DE94','Friesland':'DE94','Grafschaft Bentheim':'DE94',
    'Leer':'DE94','Oldenburg':'DE94','Oldenburg (Kreisfreie Stadt)':'DE94',
    'Osnabrück':'DE94','Osnabrück (Kreisfreie Stadt)':'DE94',
    'Wesermarsch':'DE94','Wilhelmshaven':'DE94','Wittmund':'DE94',
    'Düsseldorf':'DEA1','Duisburg':'DEA1','Essen':'DEA1','Krefeld':'DEA1',
    'Mönchengladbach':'DEA1','Mülheim an der Ruhr':'DEA1','Oberhausen':'DEA1',
    'Remscheid':'DEA1','Solingen':'DEA1','Wuppertal':'DEA1','Kleve':'DEA1',
    'Mettmann':'DEA1','Rhein-Kreis Neuss':'DEA1','Viersen':'DEA1','Wesel':'DEA1',
    'Bonn':'DEA2','Köln':'DEA2','Leverkusen':'DEA2','Städteregion Aachen':'DEA2',
    'Düren':'DEA2','Euskirchen':'DEA2','Heinsberg':'DEA2',
    'Oberbergischer Kreis':'DEA2','Rhein-Erft-Kreis':'DEA2',
    'Rhein-Sieg-Kreis':'DEA2','Rheinisch-Bergischer Kreis':'DEA2',
    'Bottrop':'DEA3','Gelsenkirchen':'DEA3','Münster':'DEA3','Borken':'DEA3',
    'Coesfeld':'DEA3','Recklinghausen':'DEA3','Steinfurt':'DEA3','Warendorf':'DEA3',
    'Bielefeld':'DEA4','Gütersloh':'DEA4','Herford':'DEA4','Höxter':'DEA4',
    'Lippe':'DEA4','Minden-Lübbecke':'DEA4','Paderborn':'DEA4',
    'Bochum':'DEA5','Dortmund':'DEA5','Hagen':'DEA5','Hamm':'DEA5','Herne':'DEA5',
    'Ennepe-Ruhr-Kreis':'DEA5','Hochsauerlandkreis':'DEA5','Märkischer Kreis':'DEA5',
    'Olpe':'DEA5','Siegen-Wittgenstein':'DEA5','Soest':'DEA5','Unna':'DEA5',
    'Ahrweiler':'DEB1','Altenkirchen (Westerwald)':'DEB1','Bad Kreuznach':'DEB1',
    'Birkenfeld':'DEB1','Cochem-Zell':'DEB1','Koblenz':'DEB1',
    'Mayen-Koblenz':'DEB1','Neuwied':'DEB1','Rhein-Hunsrück-Kreis':'DEB1',
    'Rhein-Lahn-Kreis':'DEB1','Westerwaldkreis':'DEB1',
    'Bernkastel-Wittlich':'DEB2','Eifelkreis Bitburg-Prüm':'DEB2','Trier':'DEB2',
    'Trier-Saarburg':'DEB2','Vulkaneifel':'DEB2',
    'Alzey-Worms':'DEB3','Bad Dürkheim':'DEB3','Donnersbergkreis':'DEB3',
    'Frankenthal (Pfalz)':'DEB3','Germersheim':'DEB3','Kaiserslautern':'DEB3',
    'Kaiserslautern (Kreisfreie Stadt)':'DEB3','Kusel':'DEB3',
    'Landau in der Pfalz':'DEB3','Ludwigshafen am Rhein':'DEB3','Mainz':'DEB3',
    'Mainz-Bingen':'DEB3','Neustadt an der Weinstraße':'DEB3','Pirmasens':'DEB3',
    'Rhein-Pfalz-Kreis':'DEB3','Speyer':'DEB3','Südliche Weinstraße':'DEB3',
    'Südwestpfalz':'DEB3','Worms':'DEB3','Zweibrücken':'DEB3',
    'Dresden':'DED2','Bautzen':'DED2','Görlitz':'DED2','Meißen':'DED2',
    'Sächsische Schweiz-Osterzgebirge':'DED2',
    'Chemnitz':'DED4','Erzgebirgskreis':'DED4','Mittelsachsen':'DED4',
    'Vogtlandkreis':'DED4','Zwickau':'DED4',
    'Leipzig':'DED5','Leipzig (Kreisfreie Stadt)':'DED5','Nordsachsen':'DED5',
}

# France: NAME_2 (département) → NUTS-2 (pre-2016 régions)
FR_NAME2 = {
    'Paris':'FR10','Seine-et-Marne':'FR10','Yvelines':'FR10','Essonne':'FR10',
    'Hauts-de-Seine':'FR10','Seine-Saint-Denis':'FR10','Val-de-Marne':'FR10',
    "Val-d'Oise":'FR10',
    'Cher':'FRB0','Eure-et-Loir':'FRB0','Indre':'FRB0','Indre-et-Loire':'FRB0',
    'Loir-et-Cher':'FRB0','Loiret':'FRB0',
    "Côte-d'Or":'FRC1','Nièvre':'FRC1','Saône-et-Loire':'FRC1','Yonne':'FRC1',
    'Doubs':'FRC2','Jura':'FRC2','Haute-Saône':'FRC2','Territoire de Belfort':'FRC2',
    'Calvados':'FRD1','Manche':'FRD1','Orne':'FRD1',
    'Eure':'FRD2','Seine-Maritime':'FRD2',
    'Nord':'FRE1','Pas-de-Calais':'FRE1',
    'Aisne':'FRE2','Oise':'FRE2','Somme':'FRE2',
    'Bas-Rhin':'FRF1','Haut-Rhin':'FRF1',
    'Ardennes':'FRF2','Aube':'FRF2','Marne':'FRF2','Haute-Marne':'FRF2',
    'Meurthe-et-Moselle':'FRF3','Meuse':'FRF3','Moselle':'FRF3','Vosges':'FRF3',
    'Loire-Atlantique':'FRG0','Maine-et-Loire':'FRG0','Mayenne':'FRG0',
    'Sarthe':'FRG0','Vendée':'FRG0',
    "Côtes-d'Armor":'FRH0','Finistère':'FRH0','Ille-et-Vilaine':'FRH0','Morbihan':'FRH0',
    'Dordogne':'FRI1','Gironde':'FRI1','Landes':'FRI1','Lot-et-Garonne':'FRI1',
    'Pyrénées-Atlantiques':'FRI1',
    'Corrèze':'FRI2','Creuse':'FRI2','Haute-Vienne':'FRI2',
    'Charente':'FRI3','Charente-Maritime':'FRI3','Deux-Sèvres':'FRI3','Vienne':'FRI3',
    'Aude':'FRJ1','Gard':'FRJ1','Hérault':'FRJ1','Lozère':'FRJ1','Pyrénées-Orientales':'FRJ1',
    'Ariège':'FRJ2','Aveyron':'FRJ2','Haute-Garonne':'FRJ2','Gers':'FRJ2',
    'Lot':'FRJ2','Hautes-Pyrénées':'FRJ2','Tarn':'FRJ2','Tarn-et-Garonne':'FRJ2',
    'Allier':'FRK1','Cantal':'FRK1','Haute-Loire':'FRK1','Puy-de-Dôme':'FRK1',
    'Ain':'FRK2','Ardèche':'FRK2','Drôme':'FRK2','Isère':'FRK2','Loire':'FRK2',
    'Rhône':'FRK2','Savoie':'FRK2','Haute-Savoie':'FRK2',
    'Alpes-de-Haute-Provence':'FRL0','Hautes-Alpes':'FRL0','Alpes-Maritimes':'FRL0',
    'Bouches-du-Rhône':'FRL0','Var':'FRL0','Vaucluse':'FRL0',
    'Corse-du-Sud':'FRM0','Haute-Corse':'FRM0',
}

# Italy: NAME_1 → NUTS-2 (NAME_2 for Trentino-Alto Adige)
IT_NAME1 = {
    'Abruzzo':'ITF1','Apulia':'ITF4','Basilicata':'ITF5','Calabria':'ITF6',
    'Campania':'ITF3','Emilia-Romagna':'ITH5','Friuli-Venezia Giulia':'ITH4',
    'Lazio':'ITI4','Liguria':'ITC3','Lombardia':'ITC4','Marche':'ITI3',
    'Molise':'ITF2','Piemonte':'ITC1','Sardegna':'ITG2','Sicily':'ITG1',
    'Toscana':'ITI1','Umbria':'ITI2',"Valle d'Aosta":'ITC2','Veneto':'ITH3',
}
IT_TRENTINO = {'Bolzano': 'ITH1', 'Trento': 'ITH2'}

# Spain: NAME_1 → NUTS-2 (NAME_2 for Ceuta/Melilla)
ES_NAME1 = {
    'Andalucía':'ES61','Aragón':'ES24','Cantabria':'ES13','Castilla y León':'ES41',
    'Castilla-La Mancha':'ES42','Cataluña':'ES51','Comunidad Foral de Navarra':'ES22',
    'Comunidad Valenciana':'ES52','Comunidad de Madrid':'ES30','Extremadura':'ES43',
    'Galicia':'ES11','Islas Baleares':'ES53','Islas Canarias':'ES70','La Rioja':'ES23',
    'País Vasco':'ES21','Principado de Asturias':'ES12','Región de Murcia':'ES62',
}
ES_CEUTA = {'Ceuta': 'ES63', 'Melilla': 'ES64'}

# Netherlands: NAME_1 → NUTS-2
NL_NAME1 = {
    'Drenthe':'NL13','Flevoland':'NL23','Friesland':'NL12','Gelderland':'NL22',
    'Groningen':'NL11','Limburg':'NL42','Noord-Brabant':'NL41','Noord-Holland':'NL32',
    'Overijssel':'NL21','Utrecht':'NL31','Zeeland':'NL34','Zuid-Holland':'NL33',
}

# Belgium: NAME_2 → NUTS-2
BE_NAME2 = {
    'Antwerpen':'BE21','Brabant Wallon':'BE31','Bruxelles':'BE10','Hainaut':'BE32',
    'Limburg':'BE22','Liège':'BE33','Luxembourg':'BE34','Namur':'BE35',
    'Oost-Vlaanderen':'BE23','Vlaams Brabant':'BE24','West-Vlaanderen':'BE25',
}

# UK: NAME_2 → NUTS-2
UK_NAME2 = {
    'Darlington':'UKC1','Durham':'UKC1','Hartlepool':'UKC1','Middlesbrough':'UKC1',
    'Redcar and Cleveland':'UKC1','Stockton-on-Tees':'UKC1',
    'Gateshead':'UKC2','Newcastle upon Tyne':'UKC2','North Tyneside':'UKC2',
    'Northumberland':'UKC2','South Tyneside':'UKC2','Sunderland':'UKC2',
    'Cumbria':'UKD1',
    'Bolton':'UKD3','Bury':'UKD3','Manchester':'UKD3','Oldham':'UKD3',
    'Rochdale':'UKD3','Salford':'UKD3','Stockport':'UKD3','Tameside':'UKD3',
    'Trafford':'UKD3','Wigan':'UKD3',
    'Blackburn with Darwen':'UKD4','Blackpool':'UKD4','Lancashire':'UKD4',
    'Cheshire East':'UKD6','Cheshire West and Chester':'UKD6','Halton':'UKD6','Warrington':'UKD6',
    'Knowsley':'UKD7','Saint Helens':'UKD7','Sefton':'UKD7','Wirral':'UKD7',
    'East Riding of Yorkshire':'UKE1','Kingston upon Hull':'UKE1','North Lincolnshire':'UKE1',
    'North Yorkshire':'UKE2','York':'UKE2',
    'Barnsley':'UKE3','Doncaster':'UKE3','Rotherham':'UKE3','Sheffield':'UKE3',
    'Bradford':'UKE4','Calderdale':'UKE4','Kirklees':'UKE4','Leeds':'UKE4','Wakefield':'UKE4',
    'Derby':'UKF1','Derbyshire':'UKF1','Nottingham':'UKF1','Nottinghamshire':'UKF1',
    'Leicester':'UKF2','Leicestershire':'UKF2','Northamptonshire':'UKF2','Rutland':'UKF2',
    'Lincolnshire':'UKF3',
    'Herefordshire':'UKG1','Warwickshire':'UKG1','Worcestershire':'UKG1',
    'Shropshire':'UKG2','Staffordshire':'UKG2','Stoke-on-Trent':'UKG2','Telford and Wrekin':'UKG2',
    'Birmingham':'UKG3','Coventry':'UKG3','Dudley':'UKG3','Sandwell':'UKG3',
    'Solihull':'UKG3','Walsall':'UKG3','Wolverhampton':'UKG3',
    'Cambridgeshire':'UKH1','Norfolk':'UKH1','Peterborough':'UKH1','Suffolk':'UKH1',
    'Bedfordshire':'UKH2','Central Bedfordshire':'UKH2','Hertfordshire':'UKH2','Luton':'UKH2',
    'Essex':'UKH3','Southend-on-Sea':'UKH3','Thurrock':'UKH3',
    'Bracknell Forest':'UKJ1','Buckinghamshire':'UKJ1','Milton Keynes':'UKJ1',
    'Oxfordshire':'UKJ1','Reading':'UKJ1','Slough':'UKJ1',
    'West Berkshire':'UKJ1','Windsor and Maidenhead':'UKJ1','Wokingham':'UKJ1',
    'Brighton and Hove':'UKJ2','East Sussex':'UKJ2','Surrey':'UKJ2','West Sussex':'UKJ2',
    'Hampshire':'UKJ3','Isle of Wight':'UKJ3','Portsmouth':'UKJ3','Southampton':'UKJ3',
    'Kent':'UKJ4','Medway':'UKJ4',
    'Bath and North East Somerset':'UKK1','Bristol':'UKK1','Gloucestershire':'UKK1',
    'North Somerset':'UKK1','South Gloucestershire':'UKK1','Swindon':'UKK1','Wiltshire':'UKK1',
    'Bournemouth':'UKK2','Dorset':'UKK2','Poole':'UKK2','Somerset':'UKK2','Torbay':'UKK2',
    'Cornwall':'UKK3','Isles of Scilly':'UKK3',
    'Devon':'UKK4','Plymouth':'UKK4',
    'Blaenau Gwent':'UKL1','Bridgend':'UKL1','Caerphilly':'UKL1','Carmarthenshire':'UKL1',
    'Ceredigion':'UKL1','Merthyr Tydfil':'UKL1','Neath Port Talbot':'UKL1',
    'Pembrokeshire':'UKL1','Powys':'UKL1','Rhondda, Cynon, Taff':'UKL1',
    'Swansea':'UKL1','Torfaen':'UKL1',
    'Anglesey':'UKL2','Cardiff':'UKL2','Conwy':'UKL2','Denbighshire':'UKL2',
    'Flintshire':'UKL2','Gwynedd':'UKL2','Monmouthshire':'UKL2',
    'Newport':'UKL2','Vale of Glamorgan':'UKL2','Wrexham':'UKL2',
    'Aberdeen':'UKM5','Aberdeenshire':'UKM5','Moray':'UKM5',
    'Argyll and Bute':'UKM6','Eilean Siar':'UKM6','Highland':'UKM6',
    'Orkney Islands':'UKM6','Shetland Islands':'UKM6',
    'Angus':'UKM7','Clackmannanshire':'UKM7','Dundee':'UKM7','East Lothian':'UKM7',
    'Edinburgh':'UKM7','Falkirk':'UKM7','Fife':'UKM7','Midlothian':'UKM7',
    'Perthshire and Kinross':'UKM7','Stirling':'UKM7','West Lothian':'UKM7',
    'East Ayrshire':'UKM8','East Dunbartonshire':'UKM8','East Renfrewshire':'UKM8',
    'Glasgow':'UKM8','Inverclyde':'UKM8','North Ayrshire':'UKM8',
    'North Lanarkshire':'UKM8','Renfrewshire':'UKM8','South Ayrshire':'UKM8',
    'South Lanarkshire':'UKM8','West Dunbartonshire':'UKM8',
    'Dumfries and Galloway':'UKM9','Scottish Borders':'UKM9',
    'Antrim and Newtownabbey':'UKN0','Armagh, Banbridge and Craigavon':'UKN0',
    'Belfast':'UKN0','Causeway Coast and Glens':'UKN0','Derry and Strabane':'UKN0',
    'Fermanagh and Omagh':'UKN0','Lisburn and Castlereagh':'UKN0',
    'Mid Ulster':'UKN0','Mid and East Antrim':'UKN0',
    'Newry, Mourne and Down':'UKN0','North Down and Ards':'UKN0',
    'Greater London': None,   # no single NUTS-2 in Eurostat
}

# ── Helper functions ───────────────────────────────────────────────────────

def assign_nuts2(name0, name1, name2):
    if name0 == 'Germany':
        if name1 in DE_NAME1:
            return DE_NAME1[name1]
        return DE_NAME2.get(name2)
    elif name0 == 'France':
        return FR_NAME2.get(name2)
    elif name0 == 'Italy':
        if name1 == 'Trentino-Alto Adige':
            return IT_TRENTINO.get(name2)
        return IT_NAME1.get(name1)
    elif name0 == 'Spain':
        if name1 == 'Ceuta y Melilla':
            return ES_CEUTA.get(name2)
        return ES_NAME1.get(name1)
    elif name0 == 'Netherlands':
        return NL_NAME1.get(name1)
    elif name0 == 'Belgium':
        return BE_NAME2.get(name2)
    elif name0 == 'United Kingdom':
        return UK_NAME2.get(name2)
    return None

def r2_score(y, yp):
    ss_res = np.sum((y - yp) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading pre-computed predictions ...")
veh = pd.read_csv(VEH_CSV, low_memory=False)
# Sum all vehicle types → total stock per city-year
veh['value'] = pd.to_numeric(veh['value'], errors='coerce').fillna(0)
city_total = (veh.groupby(['region', 'city', 'year'])['value']
              .sum().reset_index(name='our_total'))
print(f"  City-year-total rows: {len(city_total):,}")

print("Loading boundary data ...")
bnd = pd.read_csv(BND_CSV, low_memory=False,
                  usecols=['UID', 'NAME_0', 'NAME_1', 'NAME_2'])
bnd['nuts2_code'] = bnd.apply(
    lambda r: assign_nuts2(r['NAME_0'], r['NAME_1'], r['NAME_2']), axis=1)
uid_nuts2 = bnd.set_index('UID')['nuts2_code']
city_total['nuts2_code'] = city_total['city'].map(uid_nuts2)

print("Loading Eurostat reference data ...")
df = pd.read_csv(ESTAT_TSV, sep='\t')
first = df.columns[0]
parts = df[first].str.split(',', expand=True)
parts.columns = ['freq', 'vehicle', 'unit', 'geo']
for c in parts.columns:
    parts[c] = parts[c].str.strip()
df2 = pd.concat([parts, df.drop(columns=[first])], axis=1)
year_cols = [c for c in df2.columns if c.strip().isdigit()]
df_long = df2.melt(id_vars=['freq', 'vehicle', 'unit', 'geo'],
                   value_vars=year_cols, var_name='year', value_name='value')
df_long['year']  = df_long['year'].str.strip().astype(int)
df_long['value'] = pd.to_numeric(
    df_long['value'].astype(str).str.replace(':', '').str.strip(), errors='coerce')
estat_all = (df_long[(df_long['vehicle'] == 'CAR') & (df_long['unit'] == 'NR')
                     & df_long['value'].notna()]
             [['geo', 'year', 'value']].copy())

try:
    labels = pd.read_csv(LABELS_TSV, sep='\t', header=None,
                         names=['code', 'name']).dropna()
    code_name = labels.set_index('code')['name'].to_dict()
except Exception:
    code_name = {}

# ── Per-country validation ─────────────────────────────────────────────────
COUNTRIES = {
    'Germany':        ('DEU', (2011, 2022)),
    'France':         ('FRA', (2011, 2023)),
    'Italy':          ('ITA', (2011, 2023)),
    'Spain':          ('ESP', (2011, 2020)),
    'United Kingdom': ('GBR', (2011, 2018)),
    'Netherlands':    ('NLD', (2011, 2023)),
    'Belgium':        ('BEL', (2011, 2023)),
}

summary_rows = []
all_per_region = []

for cname, (gid0, yr_range) in COUNTRIES.items():
    print(f"\n{'='*55}")
    print(f"  {cname}")

    # City mapping coverage
    bnd_c = bnd[bnd['NAME_0'] == cname]
    total  = len(bnd_c)
    mapped = bnd_c['nuts2_code'].notna().sum()
    print(f"  Cities: {total}, mapped to NUTS-2: {mapped} ({mapped/total*100:.1f}%)")

    # Aggregate model predictions to NUTS-2 × year
    pred = (city_total[city_total['region'] == cname]
            .dropna(subset=['nuts2_code'])
            .query(f"year >= {yr_range[0]} and year <= {yr_range[1]}")
            .groupby(['nuts2_code', 'year'])['our_total']
            .sum().reset_index())

    # Eurostat reference
    nuts2_codes = bnd_c['nuts2_code'].dropna().unique()
    estat = (estat_all[estat_all['geo'].isin(nuts2_codes) &
                       estat_all['year'].between(*yr_range)]
             .rename(columns={'geo': 'nuts2_code', 'value': 'estat_total'}))
    print(f"  Eurostat: {estat['nuts2_code'].nunique()} NUTS-2 × {estat['year'].nunique()} years")

    merged = pred.merge(estat, on=['nuts2_code', 'year']).dropna()
    print(f"  Matched: {merged['nuts2_code'].nunique()} NUTS-2 × {merged['year'].nunique()} years")

    y  = merged['estat_total'].values
    yp = merged['our_total'].values
    r2_val  = r2_score(y, yp)
    mae_val = float((np.abs(yp - y) / y * 100).mean())

    print(f"\n  R²  : {r2_val:.4f}")
    print(f"  MAE : {mae_val:.1f}%")

    summary_rows.append({
        'country':  cname,
        'nuts2_n':  merged['nuts2_code'].nunique(),
        'years':    f"{yr_range[0]}–{yr_range[1]}",
        'R2':       round(r2_val, 4),
        'MAE_pct':  round(mae_val, 1),
    })

    # Per-region breakdown
    per_r = (merged.groupby('nuts2_code')
             .apply(lambda g: pd.Series({
                 'R2':  r2_score(g['estat_total'].values, g['our_total'].values),
                 'MAE': float((np.abs(g['our_total'] - g['estat_total'])
                               / g['estat_total'] * 100).mean())
             }), include_groups=False)
             .reset_index())
    per_r['name']    = per_r['nuts2_code'].map(code_name)
    per_r['country'] = cname
    all_per_region.append(per_r)

    per_r_sorted = per_r.sort_values('MAE')
    print(f"\n  Per-NUTS-2 errors (sorted by MAE):")
    print(per_r_sorted[['nuts2_code', 'name', 'R2', 'MAE']].to_string(index=False))

    # Save merged comparison
    merged['nuts2_name'] = merged['nuts2_code'].map(code_name)
    merged.to_csv(OUT_DIR / f'{gid0.lower()}_nuts2_comparison.csv', index=False)

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("SUMMARY")
summary = pd.DataFrame(summary_rows)
print(summary[['country', 'nuts2_n', 'years', 'R2', 'MAE_pct']].to_string(index=False))
summary.to_csv(OUT_DIR / 'validation_summary.csv', index=False)

all_per_region_df = pd.concat(all_per_region, ignore_index=True)
all_per_region_df.to_csv(OUT_DIR / 'per_nuts2_errors.csv', index=False)
print(f"\nOutputs saved to: {OUT_DIR}")
