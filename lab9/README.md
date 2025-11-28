b) Efectul lui Y si Theta asupra distributiei

Am observat ca rezultatele depind foarte mult de datele observate si de probabilitatea fixata.

In primul rand, cand numarul de cumparatori observati Y creste, distributia lui n se muta spre valori mai mari. Acest lucru este logic, deoarece n reprezinta numarul total de vizitatori, deci matematic n trebuie sa fie cel putin egal cu Y. Cu cat vedem mai multi cumparatori, cu atat creste limita minima a vizitatorilor posibili.

In al doilea rand, parametrul Theta influenteaza incertitudinea. Cand Theta este mic (0.2), adica probabilitatea de cumparare este mica, modelul estimeaza ca n este mult mai mare decat Y, iar distributia este mai lata (avem o incertitudine mai mare). Cand Theta este mare (0.5), modelul estimeaza ca n este destul de apropiat de Y, iar distributia este mai ingusta si mai precisa.

d) Diferenta dintre Posterior pentru n si Posterior Predictive pentru Y*

Distributia a posteriori pentru n reprezinta credinta noastra actualizata despre parametrul necunoscut, adica ne spune cati oameni au intrat probabil in magazin in ziua respectiva.

Pe de alta parte, distributia predictiva a posteriori pentru Y* reprezinta o simulare a unor posibile date viitoare. Aceasta se obtine luand valorile estimate pentru n si trecandu-le din nou prin procesul binomial. De aceea, distributia predictiva este intotdeauna mai imprastiata decat cea pentru n, deoarece cumuleaza doua tipuri de incertitudine: nesiguranta noastra legata de cat este n, plus variatia aleatoare naturala a procesului de cumparare.