# UKIDSS x WISE: Creating a Paired Dataset  
Aim to create paired dataset with UKIDSS and WISE image. 
<br/><br/>
## 1. Getting an Image at Data Archive

### 1.1 Image Size
Decide the image size considering the pixels scale of UKIDSS and Wise survey.

#### (1) Considering Pixel Scale and Search Radius
- Pixel Scale
  - UKIDSS : ~ 0.4"/pix
  - WISE  :   ~ 1.375"/pix
- Maximum Image Extraction Radius
  - UKIDSS : 15 [arcmin] (12 [arcmin] for MultiGetImage)
  - WISE: No Limits (maybe)
- Searching Radius and Image Size
  - 1[arcmin] x 1[arcmin] &rarr; UKIDSS Image Size: ~ 150x150 / WISE Image Size: ~ 44x44
  - **5[arcmin] x 5[arcmin] &rarr; UKIDSS Image Size: ~ 750x750 / WISE Image Size: ~ 220x220**
  - 10[arcmin] x 10[arcmin] &rarr; UKIDSS Image Size: ~ 1500x1500 / WISE Image Size: ~ 440x440

#### (2) Things to Keep in Mind - UKIDSS
- All images at RA=180, Dec=0 are cropped in the result of GetImage; It seems necessary to confirm the exact survey area of LAS.
- **Not all observation results were generated at the presented image size.**
- The following image shows the K-band observation results at RA=180, Dec=10.<br/>

![image](https://github.com/Jiwon-Han1/UKIDSSxWISE/assets/147721921/163123bf-77ca-41a5-80fd-4fcf78ae4e14)
<br/>
<img width="1384" alt="image" src="https://github.com/Jiwon-Han1/UKIDSSxWISE/assets/147721921/47f16136-c211-4b1d-965d-982ed5060400">
<br/>
#### (3) Things to Keep in Mind - WISE
- Following image shows the number of targets available for all filter bands.
- Decide the filter combination considering the required wavelength.
<img width="610" alt="image" src="https://github.com/Jiwon-Han1/UKIDSSxWISE/assets/147721921/97d5e50f-c2b7-4939-9db2-97d7d2afd4d7">
<br/>
<br/>

### 1.2 Image Search (Singe Image)
Extract the image at the given point with the specified search radius.

#### (1) UKIDSS
- [GetImage](http://wsa.roe.ac.uk:8080/wsa/getImage_form.jsp)
<img width="681" alt="image" src="https://github.com/Jiwon-Han1/UKIDSSxWISE/assets/147721921/7a8d2eb8-b252-4ed3-a5fd-d6d4a6c76438">


#### (2) WISE
- [Search by Position](https://irsa.ipac.caltech.edu/applications/wise/?__action=layout.showDropDown&)
<img width="696" alt="image" src="https://github.com/Jiwon-Han1/UKIDSSxWISE/assets/147721921/396148e8-9947-43fc-bf41-2e7aa39f90e6">
<br/>
<br/>

### 1.3 Making a Coordinate List

#### (1) Check the Required Format
- **UKIDSS**
  - [MultiGetImage](http://wsa.roe.ac.uk:8080/wsa/MultiGetImage_form.jsp)<br/>
  - Supply a **.txt** file of coordinates (J2000) either in decimal degrees or sexagesimal, which are separated by spaces or commas.
  - Example: <http://wsa.roe.ac.uk/examples/las.txt>  
  - <img width="241" alt="image" src="https://github.com/Jiwon-Han1/UKIDSSxWISE/assets/147721921/cf559cca-2acb-4b06-b8de-751fada2139f">

- **WISE**
  - [Search by Position (Multi-Object)](https://irsa.ipac.caltech.edu/applications/wise/?__action=layout.showDropDown&)
  - Submit the **IPAC table file**, which is ASCII text with headers explaining the type of data in each column, separated by vertical bars.
  - Detailed guide: <https://irsa.ipac.caltech.edu/onlinehelp/wise/#id=searching.byTable>
  - <img width="293" alt="image" src="https://github.com/Jiwon-Han1/UKIDSSxWISE/assets/147721921/420b630a-e848-4dce-86a7-a54a1e8a10d4">

#### (2) Create the Coordinate File




<br/><br/><br/><br/><br/><br/><br/>







## 2. Making a Targetlist

### 2.1. Check Tables
* Table Description: <http://wsa.roe.ac.uk/www/wsa_browser.html>
* Description for WISE Source Catalog: <https://wise2.ipac.caltech.edu/docs/release/prelim/expsup/sec1_4b.html>
<br/><br/>

### 2.2. Choose Table Candidates

#### UKIDSS<br/>
* lasMergeLog: <br/>
  Contains frame set details of merged LAS MultiframeDetector images. <br/>
  *> WSA_UKIDSS/Tables/lasMergeLog*<br/><br/>
* **lasSourceXwise_allskysc**: <br/>
  Cross-neighbours between las and wise_allskysc.<br/>
  *> WSA_UKIDSS/Tables/lasSourceXwise_allskysc*<br/><br/>
* **lasPointSource**: <br/>
  Merged, seamless point source catalogue derived from the UKIDSS LAS, highlighting the '*completeness*'. <br/>
  *> WSA_UKIDSS/Tables/lasPointSource*<br/><br/>
* reliableLasPointSource:<br/>
  Reliable, but incomplete, point source catalogue derived from UKIDSS LAS, highlighting the '*reliability*'. <br/>
  *> WSA_UKIDSS/Tables/reliableLasPointSource*<br/><br/>
#### WISE<br/>
* allwise_sc: <br/>
  Contains the parameters provided for each source in the AllWISE source catalogue. (DR 2013)<br/>
  *> WISE/Tables/allwise_sc* <br/><br/>
* **wise_allskysc**: <br/>
  Contains the parameters provided for each source in the WISE allsky catalogue. (DR 2012)<br/>
  *> WISE/Tables/wise_allskysc* <br/><br/>

### 2.3. Constructing SQL Queries (in UKIDSS Data Access)

#### Main Table
- **lasSourceXwise_allskysc**: <br/>
    - *masterObjID*: The unique ID in lasSource (=sourceID)
    - *slaveObjID*: The unique ID of the neighbour in WISE.wise_allskysc (=cntr)

#### Merged Table & Primary Key
- **lasPointSource**: (*or lasSource?*)<br/>
    - **sourceID** (Primary Key) : <br/>
        > UID (unique over entire WSA via programme ID prefix) of this merged detection as assigned by merge algorithm. <br/>
    - **yAperMag3 ~ kAperMag3**: <br/>
        > Default point source Y ~ K aperture corrected mag (2.0 arcsec aperture diameter)
    - yAperMag3Err ~ kAperMag3Err: <br/>
        > Error in default point source Y ~ K mag (2.0 arcsec aperture diameter) <br/>
- **wise_allskysc**: (*or allwise_sc?*) <br/>
    - **cntr** (Primary Key) : <br/>
        > Unique identification number for this object in the AllWISE Catalog/Reject Table.
    - source_id: <br/>
        > Unique source ID, formed from a combination of the Atlas Tile ID, coadd_id, and sequential extracted source number, src.
    - **w1mag ~ w4mag**: <br/> 
        > "Standard" aperture magnitude for each filter; W1 ~ W4.
        > In *allwise_sc*, this column is null if an aperture measurement was not possible.
    - w1sigm ~ w4sigm: <br/> 
        > Uncertainty in the "standard" aperture magnitude for each filter; W1 ~ W4.
        > In *allwise_sc*, this column is null if the "standard" aperture magnitude is an upper limit, or if an aperture measurement was not possible.

#### Required Characteristics
- Essential: ObjID, ra, dec, flux, etc.
- Optional: x, y coord in frame

#### Constraints (TBD)
- Brightness:
- Position & Nearby Source:

#### Matching Surveys with TargetID
Without constaints, the following query returned 55,063,366 result rows, which is same as the number of rows in *lasSourceXwise_allskysc*.
```SQL
SELECT TOP 50 Main.masterObjID AS U_ObjID, Main.slaveObjID AS W_ObjID, U.ra, U.dec, W.ra, W.dec,
       U.yAperMag3, U.yAperMag3Err, U.jAperMag3, U.jAperMag3Err, U.hAperMag3, U.hAperMag3Err, U.kAperMag3, U.kAperMag3Err, 
       W.w1mag, W.w1sigm, W.w2mag, W.w2sigm, W.w3mag, W.w3sigm, W.w4mag, W.w4sigm
  FROM lasSourceXwise_allskysc AS Main
       INNER JOIN lasPointSource AS U
       ON Main.masterObjID = U.sourceID
       INNER JOIN wise_allskysc AS W
       ON Main.slaveObjID = W.cntr
```
#### &rarr; However, it was impossible to load *wise_allskysc*. <br/> <br/>

### 2.4. Alternative Approach: Merging Tables
Instead of getting a target list at once on the UKIDSS data archive, we decided to merge the tables which are obtained from ~~~

#### (1) UKIDSS LAS Table
- Add spatial and brightness constraints considering the CrossID result.
  - **Spatial Constraints**: TBD
  - **Brightness**: 0 < kAperMag3 < 18.5 <br/>

  ![image](https://github.com/Jiwon-Han1/UKIDSSxWISE/assets/147721921/a1d78d9e-a9e4-4ece-b384-bc47f323d9d3)
  &rarr; When checking the result of CrossID at RA=180[deg] and Dec=0[deg] with in 1[arcmin] x 1[arcmin], for example, the 32 sources were detected in all filter bands and the upper limit of kAperMag3 was +18.5[mag].

```SQL
SELECT Main.masterObjID AS U_ObjID, Main.slaveObjID AS W_ObjID, U.ra, U.dec,
       U.yAperMag3, U.yAperMag3Err, U.jAperMag3, U.jAperMag3Err, U.hAperMag3, U.hAperMag3Err, U.kAperMag3, U.kAperMag3Err
  FROM lasSourceXwise_allskysc AS Main
       INNER JOIN lasPointSource AS U
       ON Main.masterObjID = U.sourceID
 WHERE U.kAperMag3 BETWEEN 0 AND 18.5
       -- (Optional): Detected Source List within 10'x10' range
       -- AND U.ra BETWEEN 180-0.083 AND 180+0.083 
       -- AND U.dec BETWEEN 0-0.083 AND 0+0.083
```
<br/><br/><br/><br/>

