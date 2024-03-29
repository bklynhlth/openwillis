###############TREMOR#################
# amptrem.praat is a Praat[6.1.47] script (http://www.praat.org/) 
# that serves as a procedure within tremor.praat.
###############TREMOR#################
# Author: Markus Brückl (markus.brueckl@tu-berlin.de)
# Copyright 2011-2021 Markus Brückl
# License: GNU GPL v3 (http://www.gnu.org/licenses/gpl.html)
######################################

######################################
# Amplitude Tremor Analysis
######################################
procedure atrem

selectObject: sndID, piID
ppID = To PointProcess (cc)
numbOfGlotPoints = Get number of points
if numbOfGlotPoints < 3
   if program_mode = 1
      beginPause: ""
         comment: "Amplitude extraction per pitch periond failed!"
         comment: "Please change arguments for Pitch extraction!"
      endPause: "Continue", 1
   endif
   if nan_out = 1
      atrm = 0
      atrc = 0
      atrf = 0
      atri = 0
      atrp = 0
      atrcip = 0
      atrps = 0
   else
      atrm = undefined
      atrc = undefined
      atrf = undefined
      atri = undefined
      atrp = undefined
      atrcip = undefined
      atrps = undefined
   endif
else
   if amplitude_extraction_method = 2
      selectObject: sndID, ppID
# amplitudes are RMS per period -- not intensity maxima ?? -- no! unclear, Praat help missing
      ampID = To AmplitudeTier (period): 0, 0, 0.0001, 0.02, 1.7
      numbOfAmpPoints = Get number of points
   elsif amplitude_extraction_method = 1
# NEW 2014-12-13: corrected misinterpretation of Praat-function "To AmplitudeTier (period)"
      numbOfAmpPoints = numbOfGlotPoints - 1
      ampID = Create AmplitudeTier: "'name$'_'name$'_'name$'", 0, slength
      for iAmpPoint to numbOfAmpPoints
         selectObject: ppID
         perStart = Get time from index: iAmpPoint
         perEnd = Get time from index: iAmpPoint+1
         selectObject: sndID
         rms = Get root-mean-square: perStart, perEnd
# very seldomly (with bad pitch settings) it occurs that perStart and perEnd are nearer 
# than sampling period -> rms would be undefined
         if rms = undefined
            samplPer = Get sampling period
            rms = Get root-mean-square: perStart-samplPer, perEnd+samplPer
         endif
         selectObject: ampID
         Add point: (perStart+perEnd)/2, rms
      endfor
   endif
######################################

if program_mode = 1
   Edit
   beginPause: "Amplitude contour"
      comment: "If there is no amplitude point or the contour is not plausible, "
      comment: "please change arguments of the initial pitch extraction!"
   endPause: "Continue", 1
endif

# since bad pitch extraction may result in not even one amplitude point
ampPointN = Get number of points
if ampPointN = 0
   atrc = undefined
   atrf = undefined
   atri = undefined
   atrp = undefined
else

# from here on out: prepare to autocorrelate AmplitudeTier-data
# sample AmplitudeTier at (constant) rate ts
# to be able to -- automatically -- read Amp. values...
   torAID = Down to TableOfReal

# to enable autocorrelation of the Amp.-contour: ->Matrix->Sound

   ma_atrem_nlcID = Create Matrix: "atrem_nlc", 0, slength, numberOfFrames+1, ts, x1, 1, 1, 1, 1, 1, "0"
# from here on out: get the mean of (the curve of) the amplitude contour in each frame
   for iframe to numberOfFrames
      selectObject: piID
      f0 = Get value in frame: iframe, "Hertz"
# determine (the time of) fixed interval borders for the resampled amplitude contour
         t = (iframe-1) * ts + x1
         tl = t - ts/2
         tu = t + ts/2
# get the indices of the amplitude points surrounding around these borders
         selectObject: ampID
         loil = Get low index from time: tl
         hiil = Get high index from time: tl
         loiu = Get low index from time: tu
         hiiu = Get high index from time: tu
# if the sound is unvoiced the amplitude is not extracted
      if f0 = undefined
         selectObject: ma_atrem_nlcID
         Set value: 1, iframe, 0
# if the amplitude contour has not begun yet...
      elsif loil = 0
         selectObject: ma_atrem_nlcID
         Set value: 1, iframe, 0
# ...or is already finished the amplitude is not extracted
      elsif hiiu = numbOfAmpPoints + 1; 
         selectObject: ma_atrem_nlcID
         Set value: 1, iframe, 0
      else
         selectObject: torAID
         lotl = Get value: loil, 1; time value of Amp.Point before tl in the PointProcess [s]
         druck_lol = Get value: loil, 2; amplitude value before tl in the PointProcess [Pa, ranged from 0 to 1]
         hitl = Get value: hiil, 1
         druck_hil = Get value: hiil, 2; amplitude value after tl in the PointProcess
         lotu = Get value: loiu, 1
         druck_lou = Get value: loiu, 2; amplitude value before tu in the PointProcess
         hitu = Get value: hiiu, 1; time value after tu in the PointProcess
         druck_hiu = Get value: hiiu, 2; amplitude value after tu in the PointProcess
# caculate (linearly interpolated) pressure/amplitude at the borders
         druck_tl = ((hitl-tl)*druck_lol + (tl-lotl)*druck_hil) / (hitl-lotl)
         druck_tu = ((hitu-tu)*druck_lou + (tu-lotu)*druck_hiu) / (hitu-lotu)

         nPinter = hiiu - 1 - loil; = loiu - loil; = hiiu - hiil; number of amp.-points between tl and tu
         if nPinter = 0; loil = loiu; hiil = hiiu
            druck_mean = (druck_tl + druck_tu) / 2
         else
            tlinter = tl
            plinter = druck_tl
            sumtdruck = 0
            for iinter from 1 to nPinter
               tuinter = Get value: loil+iinter, 1
               puinter = Get value: loil+iinter, 2
               deltat = tuinter - tlinter
               tdruck_iinter = deltat*(plinter+puinter)/2
               sumtdruck += tdruck_iinter
               tlinter = tuinter
               plinter = puinter
            endfor
            deltat = tu - tlinter
            tdruck_iinter = deltat*(plinter+druck_tu)/2
            sumtdruck += tdruck_iinter
            druck_mean = sumtdruck / ts
         endif

         selectObject: ma_atrem_nlcID
         Set value: 1, iframe, druck_mean
      endif
   endfor

# because PRAAT classifies frequencies in Pitch objects <=0 as "voiceless" and 
# therefore parts with extreme INTENSITIES would be considered as "voiceless"
# (irrelevant) after "Subtract linear fit" (1)
# "1" is added to the original Pa-values (ranged from 0 to 1) -- not to the voiceless parts
   selectObject: ma_atrem_nlcID
   for i to numberOfFrames+1
      grms =  Get value in cell: 1, i
      if grms > 0
         Set value: 1, i, grms+1
      endif
   endfor

# remove the linear amp.-trend (amplitude declination)
   pi_hlirrID = To Pitch
   Rename: "hilf_lincorr"

   pi_atremID = Subtract linear fit: "Hertz"
   Rename: "atrem"
   am_Int = Get mean: 0, 0, "Hertz"
   am_Int = am_Int - 1

# undo (1)... and normalize Amp. contour by mean Amp.
   ma_atremID = To Matrix
   for i to numberOfFrames+1
      grms =  Get value in cell: 1, i
      if grms > 0
         Set value: 1, i, (grms-1-am_Int)/am_Int
      endif
   endfor

# remove last frame, undo (2)
   ma_tremID = Create Matrix: "trem", 0, slength, numberOfFrames, ts, x1, 1, 1, 1, 1, 1, "0"
   for iframe to numberOfFrames
      selectObject: ma_atremID
      spring = Get value in cell: 1, iframe
      selectObject: ma_tremID
      Set value: 1, iframe, spring
   endfor

# to calculate autocorrelation (cc-method)
   snd_tremID = To Sound (slice): 1

# calculate Tremor Contour HNR
   call tremHNR
   if hnr = undefined and nan_out = 1
      atrHNR = 0
   else
      atrHNR = hnr
   endif

# calculate Frequency of Amplitude Tremor
   selectObject: snd_tremID
   pitrem_nID = To Pitch (cc): slength, minTr, 15, "yes", tremMagThresh, tremthresh, ocAtrem, 0.35, 0.14, maxTr
   Rename: "trem_norm"

#   atrf = Get mean: 0, 0, "Hertz"

# calculate amplitude contour magnitude and cyclicality
   call readPiO
   atrm = trm
   atrc = trc

   removeObject: pitrem_nID

# calculate Magnitude Indices of Amplitude Tremor [%]
   contType$ = "amplitude"
#   show = 0
#   call tremIntIndex
#   atri = tri
#   atrp = atri * atrf/(atrf+1)

# calculate the product of Cyclicality and Intensity Indix (at the strongest found amplitude tremor frequency)
#   atrcip = atri * atrc

# calculate (by cyclicality) weighted Sum of Intensity Indices at all found amplitude tremor frequencies
# equals the sum of ATrCIP-values for all found amplitude tremor frequencies
   call tremProdSum
   if tris = 0 and nan_out = 2
      atrps = undefined
   else
      atrps = tris
   endif
   amodN = rank

endif
endif

# clean up the Object Window
   selectObject: piID, ppID
   if numbOfGlotPoints >= 3
      plusObject: ampID
      if ampPointN > 0
         plusObject: torAID
         plusObject: ma_atrem_nlcID
         plusObject: pi_hlirrID
         plusObject: pi_atremID
         plusObject: ma_atremID
         plusObject: ma_tremID
         plusObject: snd_tremID
         plusObject: torID
      endif
   endif
   Remove
endproc