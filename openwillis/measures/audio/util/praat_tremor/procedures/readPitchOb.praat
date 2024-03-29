###############TREMOR#################
# readPitchOb.praat is a Praat[6.1.47] script (http://www.praat.org/) 
# that serves as a procedure within tremor.praat.
###############TREMOR#################
# Author: Markus Brückl (markus.brueckl@tu-berlin.de)
# Copyright 2011-2021 Markus Brückl
# License: GNU GPL v3 (http://www.gnu.org/licenses/gpl.html)
######################################

######################################
# Read from 1 FRAME (!) Praat Pitch objects
######################################

procedure readPiO

Save as text file: "./temp"
tempID = Read Strings from raw text file: "./temp"

stringN = Get number of strings
sN = 0
freq = 100
for istring from 10 to stringN
   selectObject: tempID
   tEkst$ = Get string: istring
   tEkst$ = replace$(tEkst$, " ", "",100)

   if startsWith(tEkst$, "intensity")
      trm = extractNumber(tEkst$, "intensity=")
   elsif startsWith(tEkst$, "nCandidates")
      cN = extractNumber(tEkst$, "nCandidates=")
      torID = Create TableOfReal: "Pitch_trem_norm", cN, 3
      Set column label (index): 1, "origPos"
      Set column label (index): 2, "frequency"
      Set column label (index): 3, "strength"
   elsif startsWith(tEkst$, "frequency")
      freq = extractNumber(tEkst$, "frequency=")
   elsif startsWith(tEkst$, "strength") and (freq <= maxTr)
      sN +=1
      strength = extractNumber(tEkst$, "strength=")
      selectObject: torID
      Set value: sN, 1, sN
      Set value: sN, 2, freq
# "-" because of 'Sort by column' sorts greater values down 
      Set value: sN, 3, -strength
   endif
endfor

selectObject: torID
Sort by column: 3, 0
# corrects the negation due to sorting
for ipitch from 1 to sN
   neg = Get value: ipitch, 3
   Set value: ipitch, 3, -neg
endfor

selectObject: torID
trc = Get value: 1, 3

removeObject: tempID
deleteFile: "./temp"

endproc