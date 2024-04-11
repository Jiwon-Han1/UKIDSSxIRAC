# UKIDSS x WISE: Creating a Paired Dataset  
Aim to create paired dataset with UKIDSS and WISE image. 
<br/><br/>
## 1. Making a Targetlist

### 1.1. Check Tables
* Table Description: <http://wsa.roe.ac.uk/www/wsa_browser.html>
* Description for WISE Source Catalog: <https://wise2.ipac.caltech.edu/docs/release/prelim/expsup/sec1_4b.html>
<br/><br/>

### 1.2. Choose Table Candidates

#### UKIDSS<br/>
* **lasMergeLog**: <br/>
  Contains frame set details of merged LAS MultiframeDetector images. <br/>
  *> WSA_UKIDSS/Tables/lasMergeLog*<br/><br/>
* **lasSourceXwise_allskysc**: <br/>
  Cross-neighbours between las and wise_allskysc.<br/>
  *> WSA_UKIDSS/Tables/lasSourceXwise_allskysc*<br/><br/>
* **lasPointSource**: <br/>
  Merged, seamless point source catalogue derived from the UKIDSS LAS, highlighting the '*completeness*'. <br/>
  *> WSA_UKIDSS/Tables/lasPointSource*<br/><br/>
* **reliableLasPointSource**:<br/>
  Reliable, but incomplete, point source catalogue derived from UKIDSS LAS, highlighting the '*reliability*'. <br/>
  *> WSA_UKIDSS/Tables/reliableLasPointSource*<br/><br/>
#### WISE<br/>
* **allwise_sc**: <br/>
  Contains the parameters provided for each source in the AllWISE source catalogue. (DR 2013)<br/>
  *> WISE/Tables/allwise_sc* <br/><br/>
* **wise_allskysc**: <br/>
  Contains the parameters provided for each source in the WISE allsky catalogue. (DR 2012)<br/>
  *> WISE/Tables/wise_allskysc* <br/><br/>

### 1.3. Constructing SQL Queries

#### Main Table
- **lasSourceXwise_allskysc**: <br/>
    - *masterObjID*: The unique ID in lasSource (=sourceID)
    - *slaveObjID*: The unique ID of the neighbour in WISE.wise_allskysc (=cntr)

#### Merged Table & Primary Key
- **allwise_sc**: <br/>
    - *cntr* (PK): <br/>
        > Unique identification number for this object in the AllWISE Catalog/Reject Table.
    - source_id: <br/>
        > Unique source ID, formed from a combination of the Atlas Tile ID, coadd_id, and sequential extracted source number, src.
<br/>
- **lasPointSource**: (*or ~~lasSource~~?*)<br/>
    - *sourceID* (PK): <br/>
        > UID (unique over entire WSA via programme ID prefix) of this merged detection as assigned by merge algorithm.

#### Required Characteristics
- ObjID:

#### Matching Surveys with TargetID
```SQL
SELECT Main.masterObjID AS U_ObjID, slaveObjID AS W_ObjID
  FROM lasSourceXwise_allskysc AS Main
  INNER JOIN ~~~ AS U
  ON Main.masterObjID = U.~~~
  INNER JOIN allwise_sc AS W
  ON Main.slaveObjID = W.cntr
WHERE 
```
