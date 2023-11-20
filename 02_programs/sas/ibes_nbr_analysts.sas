/* IBES: number of analysts 
*/

data a_funda (keep = key gvkey fyear datadate conm);
set comp.funda;
/* create key to uniquely identify firm-year */
key = gvkey || fyear; 
/* general filter to drop doubles from Compustat Funda */
if indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C' ;
run;
 
/* get permno */
proc sql;
  create table b_permno as
  select a.*, b.lpermno as permno
  from a_funda a left join crsp.ccmxpf_linktable b
    on a.gvkey = b.gvkey
    and b.lpermno ne .
    and b.linktype in ("LC" "LN" "LU" "LX" "LD" "LS")
    and b.linkprim IN ("C", "P") 
    and ((a.datadate >= b.LINKDT) or b.LINKDT = .B) and 
       ((a.datadate <= b.LINKENDDT) or b.LINKENDDT = .E)   ;
quit;

/* retrieve historic cusip */
proc sql;
  create table c_cusip as
  select a.*, b.ncusip
  from b_permno a, crsp.dsenames b
  where 
        a.permno = b.PERMNO
    and b.namedt <= a.datadate <= b.nameendt
    and b.ncusip ne "";
  quit;
 
/* force unique records */
proc sort data=c_cusip nodupkey; by key;run;
 
/* get ibes ticker */
proc sql;
  create table d_ibestick as
  select distinct a.*, b.ticker as ibes_ticker
  from c_cusip a, ibes.idsum b
  where 
        a.NCUSIP = b.CUSIP
    and a.datadate > b.SDATES 
;
quit;
 
/* get number of estimates -- last month of fiscal year*/
proc sql;
  create table e_numanalysts as
  select a.*, b.STATPERS, b.numest as num_analysts
  from d_ibestick a, ibes.STATSUMU_EPSUS b
  where 
        a.ibes_ticker = b.ticker
    and b.MEASURE="EPS"
    and b.FISCALP="ANN"
    and b.FPI = "1"
    and a.datadate - 30 < b.STATPERS < a.datadate 
    and a.datadate -5 <= b.FPEDATS <= a.datadate +5
;
quit;
 
/* force unique records */
proc sort data=e_numanalysts nodupkey; by key;run;
 
/* append num_analysts to b_permno */
proc sql;
    create table f_funda_analysts as 
    select a.*, b.num_analysts 
    from b_permno a 
    left join e_numanalysts b 
    on a.key=b.key;
quit;
 
/* missing num_analysts means no analysts following */
data f_funda_analysts;
set f_funda_analysts;
if permno ne . and num_analysts eq . then num_analysts = 0;
run;