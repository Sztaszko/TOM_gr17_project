# TOM Grupa 17

Projekt realizowany w ramach przedmiotu Techniki Obrazowania Medycznego 2019/2020

Plan realizacji projektu grupy 17

Skład grupy (imię, nazwisko, nr indeksu):

Angelika Kiełbasa 296995

Maja Kałwak 296993

Stanisław Kaczmarski 296992

Poniżej przedstawiono punktowy plan realizacji projektu. Wybranym algorytmem
segmentującym jest sieć neuronowa U-net 2D ​ ze względu na jej pierwotny cel segmentacji
obrazów medycznych, szeroką literaturę oraz bardzo dobre wyniki uzyskiwane na danych
KiTS19.

1. Charakterystyka zbioru danych - podział na zbiór treningowy i testowy, format
danych, wymiary zdjęć itd. ​ 
2. Implementacja architektury U-net i odpowiedniego preprocessingu ​ 
3. Dobór odpowiednich parametrów sieci ​ 
4. Implementacja walidacji wyników (DICE-score) ​ 
5. Dobór sposobu wizualizacji uzyskanych wyników
6. Bugfix
7. Napisanie i oddanie raportu ​ 

Podział obowiązków:

Angelika Kiełbasa - research preprocessingu, implementacja architektury sieci, wizualizacja
uzyskanych wyników

Maja Kałwak - opis danych, nadzór systemu kontroli wersji, implementacja architektury sieci

Stanisław Kaczmarski - implementacja architektury
implementacja ewaluacji wyników
sieci, nadzór merge-request,

Podział obowiązków może ulec zmianie w zależności od napotkanej złożoności problemu oraz
od wymaganych potrzeb.
