###############TREMOR#################
# tremIntIndex.praat is a Praat[6.1.47] script (http://www.praat.org/) 
# that calculates Tremor Intensity Indices [%] within tremor.praat.
###############TREMOR#################
# Author: Markus Brückl (markus.brueckl@tu-berlin.de)
# Copyright 2011-2021 Markus Brückl
# License: GNU GPL v3 (http://www.gnu.org/licenses/gpl.html)
######################################

procedure tremIntIndex
   selectObject: snd_tremID, pitrem_normID
   ppMaxID = To PointProcess (peaks): "yes", "no"
   Rename: "Maxima"
   numberofMaxPoints = Get number of points
   tri_max = 0
   noFMax = 0
   for iPoint to numberofMaxPoints
      selectObject: ppMaxID
      ti = Get time from index: iPoint
      selectObject: snd_tremID
      tri_Point = Get value at time: "Average", ti, "Sinc70"
      if tri_Point = undefined
         tri_Point = 0
         noFMax += 1
      endif
      tri_max += abs(tri_Point)
   endfor

if program_mode = 1
   selectObject: snd_tremID, ppMaxID
   Edit
   beginPause: ""
      comment: "Normalized and de-declined 'contType$' contour and"
      comment: "maxima of its 'rank'. strongest ('trStren:2') modulation at 'trFreq:2'Hz"
   endPause: "Continue", 1
endif
   
# tri_max:= (mean) procentual deviation of contour maxima from mean contour at trf
   numberofMaxima = numberofMaxPoints - noFMax
   tri_max = 100 * tri_max/numberofMaxima

   selectObject: snd_tremID, pitrem_normID
   ppMinID = To PointProcess (peaks): "no", "yes"
   Rename: "Minima"
   numberofMinPoints = Get number of points
   tri_min = 0
   noFMin = 0
   for iPoint from 1 to numberofMinPoints
      selectObject: ppMinID
      ti = Get time from index... iPoint
      selectObject: snd_tremID
      tri_Point = Get value at time: "Average", ti, "Sinc70"
      if tri_Point = undefined
         tri_Point = 0
         noFMin += 1
      endif
      tri_min += abs(tri_Point)
   endfor

if program_mode = 1
   selectObject: snd_tremID, ppMinID
   Edit
   beginPause: ""
      comment: "Normalized and de-declined 'contType$' contour and"
      comment: "minima of its 'rank'. strongest ('trStren:2') modulation at 'trFreq:2'Hz"
   endPause: "Continue", 1
endif

# tri_min:= (mean) procentual deviation of contour minima from mean contour at trf
   numberofMinima = numberofMinPoints - noFMin
   tri_min = 100 * tri_min/numberofMinima

   tri = (tri_max + tri_min) / 2

   removeObject:  pitrem_normID, ppMaxID, ppMinID
   
endproc