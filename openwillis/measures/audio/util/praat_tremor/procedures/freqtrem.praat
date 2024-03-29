###############TREMOR#################
# freqtrem.praat is a Praat[6.1.47] script (http://www.praat.org/) 
# that serves as a procedure within tremor.praat.
###############TREMOR#################
# Author: Markus Brückl (markus.brueckl@tu-berlin.de)
# Copyright 2011-2021 Markus Brückl
# License: GNU GPL v3 (http://www.gnu.org/licenses/gpl.html)
######################################

######################################
# Frequency Tremor Analysis
######################################
procedure ftrem
   piID = To Pitch (cc): ts, minPi, 15, "yes", silThresh, voiThresh, ocCo, ocjCo, vuvCo, maxPi

if program_mode = 1
   Edit
   beginPause: "Fundamental frequency contour"
      comment: "If the frequency contour is not properly extracted,"
      comment: "all tremor measurements will be unreliable!"
   endPause: "Continue", 1
endif

   numberVoice = Count voiced frames
if numberVoice = 0
   if program_mode = 1
      pause Pitch extraction failed! Please change arguments for Pitch extraction!
   endif
   if nan_out = 1
      ftrm = 0
      ftrc = 0
      ftrf = 0
      ftri = 0
      ftrp = 0
      ftrcip = 0
      ftrps = 0
   else
      ftrm = undefined
      ftrc = undefined
      ftrf = undefined
      ftri = undefined
      ftrp = undefined
      ftrcip = undefined
      ftrps = undefined
   endif
else

# because PRAAT only runs "Subtract linear fit" if the last frame is "voiceless" (!?):
# numberOfFrames+1 (1)
   numberOfFrames = Get number of frames
   x1 = Get time from frame number: 1
#   am_F0 = Get mean: 0, 0, "Hertz"

   ma_ftrem_0ID = Create Matrix: "ftrem_0", 0, slength, numberOfFrames+1, ts, x1, 1, 1, 1, 1, 1, "0"
   for i to numberOfFrames
      selectObject: piID
      f0 = Get value in frame: i, "Hertz"
      selectObject: ma_ftrem_0ID
# write zeros to matrix where frames are voiceless
      if f0 = undefined
         Set value: 1, i, 0
      else
         Set value: 1, i, f0
      endif
   endfor

# remove the linear F0 trend (F0 declination)
   pi_ftrem_0ID = To Pitch
   pi_ftrem_0_linID = Subtract linear fit: "Hertz"
   Rename: "ftrem_0_lin"

# undo (1)
   ma_tremID = Create Matrix: "trem", 0, slength, numberOfFrames, ts, x1, 1, 1, 1, 1, 1, "0"
   for i to numberOfFrames
      selectObject: pi_ftrem_0_linID
      f0 = Get value in frame: i, "Hertz"
      selectObject: ma_tremID
# write zeros to matrix where frames are voiceless
      if f0 = undefined
         Set value: 1, i, 0
      else
         Set value: 1, i, f0
      endif
   endfor

   pi_tremID = To Pitch
   am_F0 = Get mean: 0, 0, "Hertz"

# normalize F0-contour by mean F0
   selectObject: ma_tremID
   Formula: "(self-am_F0)/am_F0"

# since zeros in the Matrix (unvoiced frames) become normalized to -1 but 
# unvoiced frames should be zero (if anything)
# write zeros to matrix where frames are voiceless
   for i from 1 to numberOfFrames
      selectObject: pi_tremID
      f0 = Get value in frame: i, "Hertz"
      if f0 = undefined
         selectObject: ma_tremID
         Set value: 1, i, 0
      endif
   endfor

# to calculate autocorrelation (cc-method):
   selectObject: ma_tremID
   snd_tremID = To Sound (slice): 1

# calculate Tremor Contour HNR
   call tremHNR
   if hnr = undefined and nan_out = 1
      ftrHNR = 0
   else
      ftrHNR = hnr
   endif

# calculate Frequency of Frequency Tremor [Hz]
   selectObject: snd_tremID
   pitrem_nID = To Pitch (cc): slength, minTr, 15, "yes", tremMagThresh, tremthresh, ocFtrem, 0.35, 0.14, maxTr
   Rename: "trem_norm"

#   ftrf = Get mean: 0, 0, "Hertz"

# calculate frequency contour magnitude and cyclicality
   call readPiO
   ftrm = trm
   ftrc = trc 
   
   removeObject: pitrem_nID

# calculate Magnitude Indices of Frequency Tremor [%]
   contType$ = "frequency"
#   show = 0
#   call tremIntIndex
#   ftri = tri
#   ftrp = ftri * ftrf/(ftrf+1)

# calculate the product of Cyclicality and Intensity Indix (at the strongest found frequency tremor frequency)
#   ftrcip = ftri * ftrc

# calculate (by cyclicality) weighted Sum of Intensity Indices at all found frequency tremor frequencies
# equals the sum of FTrCIP-values for all found frequency tremor frequencies
   call tremProdSum
   if tris = 0 and nan_out = 2
      ftrps = undefined
   else
      ftrps = tris
   endif
   fmodN = rank

# clean up the Object Window
   removeObject: ma_ftrem_0ID, pi_ftrem_0ID, pi_ftrem_0_linID, ma_tremID, 
              ...pi_tremID, snd_tremID, torID

endif
endproc