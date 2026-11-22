# Error analysis (exact match; up to 8 examples per type)


## heuristics on `dataturks`

**Name** - 0 false positives, 1 false negatives
- FP (predicted, not in gold): 
- FN (in gold, missed):       `prashanth badala`

**Company** - 89 false positives, 195 false negatives
- FP (predicted, not in gold): `education b.s information systems`; `iot engineer maveric systems`; `microsoft corporation`; `technologies`; `ecllipse web technologies`; `mazenet solutions`; `operating systems`; `sprint value added services`
- FN (in gold, missed):       `goto.com.pk`; `nayatel`; `trg pakistan`; `snowflake`; `skava an infosys company`; `fortinet`; `oracle`; `rapid7`

**Designation** - 147 false positives, 101 false negatives
- FP (predicted, not in gold): `engineer`; `engineer`; `engineer`; `lead`; `technical lead`; `qa engineer`; `application manager`; `intern`
- FN (in gold, missed):       `ai engineer`; `iot engineer`; `it support`; `noc and qa engineer`; `configuration engineer`; `server support engineer`; `performance engineer`; `game developer`

**Location** - 567 false positives, 219 false negatives
- FP (predicted, not in gold): `python java`; `docker kubernetes`; `python java`; `docker kubernetes`; `python java`; `docker kubernetes`; `python java`; `docker kubernetes`
- FN (in gold, missed):       `amsterdam`; `hyderabad`; `san francisco`; `multan`; `lahore bund road`; `sulthan bathery`; `coimbatore`; `atlanta`

**College** - 43 false positives, 153 false negatives
- FP (predicted, not in gold): `oracle university`; `vivekananda engineering college`; `tejaa shakthi institute of technology`; `kongu vellalar mat hr sec school`; `did compering in college`; `b.tech from annamacharya institute of technology`; `university of texas`; `punjab technical university`
- FN (in gold, missed):       `uet peshawar`; `uop`; `qau`; `pucit`; `vivekananda engineering college for women`; `tejaa shakthi institute of technology for womens`; `georgia tech`; `muet`

**Degree** - 192 false positives, 220 false negatives
- FP (predicted, not in gold): `m.sc`; `m.s`; `ph.d`; `b.s`; `m.s`; `master`; `b.e`; `m.s`
- FN (in gold, missed):       `m.sc data analytics`; `m.s cybersecurity`; `ph.d artificial intelligence`; `b.s information systems`; `engineering director`; `m.s cloud computing`; `m.s software engineering`; `b.tech`


## base on `dataturks`

**Name** - 156 false positives, 120 false negatives
- FP (predicted, not in gold): `fat`; `ali`; `shrini`; `dhi se`; `va kumar`; `kumar`; `ara khan`; `ko`
- FN (in gold, missed):       `fatima ali`; `anum fatima`; `soumya balan`; `shrinidhi selva kumar`; `zara khan`; `prashanth badala`; `mohini gupta`; `sahar ali`

**Company** - 1463 false positives, 41 false negatives
- FP (predicted, not in gold): `goto`; `com`; `uet peshawar`; `ku`; `net`; `na`; `yatel`; `ku`
- FN (in gold, missed):       `goto.com.pk`; `trg pakistan`; `infosys offshore lead`; `infosys pune maharashtra`; `mall91 pakistan`; `cheetay.pk`; `rapid7`; `founding years learning solutions`

**Designation** - 0 false positives, 203 false negatives
- FP (predicted, not in gold): 
- FN (in gold, missed):       `senior developer`; `ai engineer`; `backend developer`; `senior software engineer`; `iot engineer`; `it support`; `noc and qa engineer`; `devops engineer`

**Location** - 293 false positives, 31 false negatives
- FP (predicted, not in gold): `pakistan`; `snowflake`; `mu`; `lta`; `snowf`; `than bathery`; `kerala`; `bangalore`
- FN (in gold, missed):       `multan`; `sulthan bathery`; `faisalabad`; `islamabad f 7`; `irinchayam b.o`; `hyderabad`; `multan`; `sialkot`

**College** - 0 false positives, 235 false negatives
- FP (predicted, not in gold): 
- FN (in gold, missed):       `uet peshawar`; `uop`; `qau`; `pucit`; `ned university of engineering`; `vivekananda engineering college for women`; `frankfinn institute of airhostess training`; `tejaa shakthi institute of technology for womens`

**Degree** - 0 false positives, 246 false negatives
- FP (predicted, not in gold): 
- FN (in gold, missed):       `m.sc data analytics`; `m.s cybersecurity`; `ph.d artificial intelligence`; `b.s information systems`; `bachelors in computer applications`; `be in computer science and engineering`; `b.e in cse`; `engineering director`


## finetuned on `dataturks`

**Name** - 1 false positives, 0 false negatives
- FP (predicted, not in gold): `lammal`
- FN (in gold, missed):       

**Company** - 14 false positives, 22 false negatives
- FP (predicted, not in gold): `microsoft`; `infosys`; `info`; `sy`; `microsoft office`; `microsoft`; `cisco`; `fidelity investments`
- FN (in gold, missed):       `skava an infosys company`; `oracle`; `microsoft`; `infosys offshore lead`; `infosys limited`; `infosys pune maharashtra`; `founding years learning solutions`; `microsoft`

**Designation** - 19 false positives, 19 false negatives
- FP (predicted, not in gold): `technical support engineer`; `technical`; `no`; `co`; `senior associate`; `associate consultant`; `hr`; `hr trainee`
- FN (in gold, missed):       `noc and qa engineer`; `devops engineer`; `cloud engineer`; `co coordinator of code hunter at cgc fest`; `senior associate consultant`; `hr trainee associate`; `hr executive`; `microsoft office suite expert`

**Location** - 5 false positives, 9 false negatives
- FP (predicted, not in gold): `inchayam b`; `velammal`; `uru`; `hyderabad`; `jalhandar`
- FN (in gold, missed):       `coimbatore`; `hyderabad`; `irinchayam b.o`; `hyderabad`; `chennai`; `hyderabad`; `pune`; `indeed.com r asish ratha 853988e0e0e236a3`

**College** - 26 false positives, 15 false negatives
- FP (predicted, not in gold): `hi institute of technology for womens`; `kongu vellalar mat hr sec school`; `ntu`; `annamacharya institute of technology jntu`; `punjab technical university`; `vidyapeeth university`; `goa institute of management`; `uttar pradesh technical university`
- FN (in gold, missed):       `tejaa shakthi institute of technology for womens`; `bharati vidyapeeth university pune maharashtra`; `crescent public school new delhi delhi`; `cdac acts`; `government engineering college bhuj`; `university college of engineering`; `c v raman university bilaspur`; `ignou`

**Degree** - 20 false positives, 13 false negatives
- FP (predicted, not in gold): `be in computer science`; `engineering`; `state`; `b.tech b.e`; `bc`; `pgdm in marketing`; `b.tech in computer science and engineering`; `s.c`
- FN (in gold, missed):       `be in computer science and engineering`; `bca`; `bachelor of engineering in digital marketing`; `bachelor of technology in computer science`; `secondary school certificate in design pattern`; `computer networking`; `state board`; `b.b.a in uit`


## heuristics on `synthetic`

**Name** - 1 false positives, 26 false negatives
- FP (predicted, not in gold): `washington dc`
- FN (in gold, missed):       `priya singh`; `liam o connor`; `chen wei`; `sarah connor`; `bilal ahmed`; `ayesha malik`; `hamza ali`; `zainab fatima`

**Company** - 0 false positives, 30 false negatives
- FP (predicted, not in gold): 
- FN (in gold, missed):       `oracle`; `ibm`; `amazon`; `deloitte`; `barclays`; `tencent`; `alibaba`; `booz allen hamilton`

**Designation** - 12 false positives, 13 false negatives
- FP (predicted, not in gold): `engineer`; `analyst`; `consultant`; `data lead`; `developer`; `designer`; `engineer`; `director`
- FN (in gold, missed):       `ui engineer`; `cybersecurity analyst`; `security consultant`; `big data lead`; `mern stack developer`; `vp of engineering`; `ui designer`; `site reliability engineer`

**Location** - 10 false positives, 26 false negatives
- FP (predicted, not in gold): `kafka kubernetes`; `computer science king`; `react redux`; `software engineering peking university`; `penetration testing wireshark`; `metasploit linux`; `agile scrum`; `azure angular`
- FN (in gold, missed):       `shenzhen`; `hangzhou`; `washington dc`; `stockholm`; `lahore`; `karachi`; `lahore`; `islamabad`

**College** - 15 false positives, 26 false negatives
- FP (predicted, not in gold): `king s college`; `pearls karachi ms software engineering ned university`; `bs in computer science from university of california`; `bs in information technology from stanford university`; `master of computer science from university of illinois`; `bachelor of science from university of british columbia`; `he holds an mba from wharton school`; `fine arts from rhode island school of design`
- FN (in gold, missed):       `georgia tech`; `king s college london`; `mit`; `usc`; `fast nuces`; `ned university`; `uet lahore`; `nust`

**Degree** - 18 false positives, 26 false negatives
- FP (predicted, not in gold): `phd`; `b.sc`; `bachelor`; `bs in computer science from university`; `bs in information technology from stanford`; `master of computer science from university of illinois`; `bachelor of science from university of british columbia`; `master of fine arts from rhode island school of design`
- FN (in gold, missed):       `bs computer science`; `ms artificial intelligence`; `phd data science`; `b.sc computer science`; `bs cs`; `ms data analytics`; `bs cs`; `ms software engineering`


## base on `synthetic`

**Name** - 16 false positives, 9 false negatives
- FP (predicted, not in gold): `riya singh`; `bo`; `oz allen hamilton`; `le`; `scrum`; `ji`; `ilal ahmed`; `aye`
- FN (in gold, missed):       `priya singh`; `liam o connor`; `sarah connor`; `bilal ahmed`; `ayesha malik`; `hamza ali`; `zainab fatima`; `rahul verma`

**Company** - 76 false positives, 12 false negatives
- FP (predicted, not in gold): `microservices`; `spring boot`; `dock`; `kafka`; `ku`; `netes`; `aws`; `georgia tech`
- FN (in gold, missed):       `amazon`; `deloitte`; `booz allen hamilton`; `google`; `spotify`; `10pearls`; `netsol technologies`; `careem`

**Designation** - 0 false positives, 38 false negatives
- FP (predicted, not in gold): 
- FN (in gold, missed):       `senior java developer`; `software engineer`; `data scientist`; `analyst`; `devops lead`; `devops engineer`; `frontend developer`; `ui engineer`

**Location** - 7 false positives, 5 false negatives
- FP (predicted, not in gold): `amazon`; `seattle`; `wa`; `london`; `uk`; `care`; `ip`
- FN (in gold, missed):       `seattle wa`; `london uk`; `karachi`; `lahore`; `lahore`

**College** - 0 false positives, 31 false negatives
- FP (predicted, not in gold): 
- FN (in gold, missed):       `georgia tech`; `stanford university`; `university of washington`; `king s college london`; `peking university`; `george mason university`; `mit`; `usc`

**Degree** - 0 false positives, 31 false negatives
- FP (predicted, not in gold): 
- FN (in gold, missed):       `bs computer science`; `ms artificial intelligence`; `phd data science`; `b.sc computer science`; `bachelor of software engineering`; `master of cybersecurity`; `bs cs`; `ms data analytics`


## finetuned on `synthetic`

**Name** - 22 false positives, 10 false negatives
- FP (predicted, not in gold): `boo`; `allen hamilton`; `django`; `mark stone is`; `she`; `david hall is`; `she`; `james wilson is`
- FN (in gold, missed):       `mark stone`; `david hall`; `james wilson`; `jonathan pierce`; `sarah goldberg`; `rahul verma`; `carlos mendez`; `lucas silva`

**Company** - 9 false positives, 31 false negatives
- FP (predicted, not in gold): `limited`; `pe`; `nets`; `in`; `in`; `services`; `force`; `coinbase in`
- FN (in gold, missed):       `oracle`; `ibm`; `deloitte`; `barclays`; `tencent`; `alibaba`; `booz allen hamilton`; `raytheon`

**Designation** - 15 false positives, 19 false negatives
- FP (predicted, not in gold): `end developer`; `he`; `administrator`; `he`; `serves`; `he holds`; `is`; `he`
- FN (in gold, missed):       `analyst`; `devops lead`; `frontend developer`; `ui engineer`; `software engineer`; `big data lead`; `product manager`; `mern stack developer`

**Location** - 8 false positives, 21 false negatives
- FP (predicted, not in gold): `seattle`; `london`; `hang`; `seattle`; `dublin`; `miami`; `pune`; `stockholm`
- FN (in gold, missed):       `seattle wa`; `london uk`; `hangzhou`; `stockholm`; `san francisco`; `new york`; `san jose`; `chicago illinois`

**College** - 7 false positives, 6 false negatives
- FP (predicted, not in gold): `of`; `from`; `university of`; `british columbia`; `from`; `from`; `from`
- FN (in gold, missed):       `mit`; `usc`; `stockholm school of economics`; `fast nuces`; `pucit`; `university of british columbia`

**Degree** - 13 false positives, 13 false negatives
- FP (predicted, not in gold): `phd`; `science`; `bs cs mit`; `bs`; `bs in computer science from`; `graduated`; `earned`; `bs in information technology from`
- FN (in gold, missed):       `bs computer science`; `ms artificial intelligence`; `phd data science`; `bachelor of software engineering`; `bs cs`; `ms data analytics`; `mba`; `bs cs`

