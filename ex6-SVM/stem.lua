--[[
   This software is based on the Porter Stemmer which can be found here:
   http://tartarus.org/~martin/PorterStemmer/

   The general algorithm is based on the Javascript version from the above page
   (by 'Andargor'?) and then verified against the C reference source.

   MIT LICENSE

   Copyright (C) 2012 Chris Osgood <chris at luadev.com>
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of  this  software  and  associated documentation files (the "Software"), to
   deal  in  the Software without restriction, including without limitation the
   rights  to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell  copies  of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
  
   The  above  copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
  
   THE  SOFTWARE  IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED,  INCLUDING  BUT  NOT  LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS  FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS  OR  COPYRIGHT  HOLDERS  BE  LIABLE  FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY,  WHETHER  IN  AN  ACTION  OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE. 
--]]
module("stem", package.seeall)

require("lpeg")

-- Makes pattern match at end of string
local function eos(p)
   return (1 - (p * -1))^0 * lpeg.C(p) * -1
end

-- Word patterns
local c = 1 - lpeg.S("aeiou")   -- consonant
local v = lpeg.S("aeiouy")      -- vowel
local C = c * (1 - v)^0         -- consonant sequence
local V = v * lpeg.S("aeiou")^0 -- vowel sequence

local mgr0 = C^-1 * V * C             -- [C]VC... is m>0
local meq1 = C^-1 * V * C * V^-1 * -1 -- [C]VC[V] is m=1
local mgr1 = C^-1 * V * C * V * C     -- [C]VCVC... is m>1
local s_v  = C^-1 * v                 -- vowel in stem
local e_v  = C * v * (1 - lpeg.S("aeiouwxy")) * -1

-- Suffix patterns
local step2map = {
   ["ational"] = "ate", ["ization"] = "ize", ["iveness"] = "ive",
   ["fulness"] = "ful", ["ousness"] = "ous", ["tional"] = "tion",
   ["biliti"] = "ble",  ["entli"] = "ent",   ["ousli"] = "ous",
   ["ation"] = "ate",   ["alism"] = "al",    ["aliti"] = "al",
   ["iviti"] = "ive",   ["enci"] = "ence",   ["anci"] = "ance",
   ["izer"] = "ize",    ["alli"] = "al",     ["ator"] = "ate",
   ["logi"] = "log",    ["bli"] = "ble",     ["eli"] = "e"
}

local step3map = {
   ["icate"] = "ic", ["ative"] = "",  ["alize"] = "al",
   ["iciti"] = "ic", ["ical"] = "ic", ["ful"] = "",
   ["ness"] = ""
}

local step1a1 = eos(lpeg.P("sses") + "ies")
local step1a2 = eos((1 - lpeg.P("s")) * "s")
local step1b1 = eos(lpeg.P("eed"))
local step1b2 = eos(lpeg.P("ed") + "ing")
local step1b3 = eos(lpeg.P("at") + "bl" + "iz")
local step1c  = eos(lpeg.P("y"))

local step2 = eos(lpeg.P("ational") + "ization" + "iveness" + "fulness" +
                  "ousness" + "tional" + "biliti" + "entli" + "ousli" +
                  "ation" + "alism" + "aliti" + "iviti" + "enci" + "anci" +
                  "izer" + "alli" + "ator" + "logi" + "bli" + "eli")

local step3 = eos(lpeg.P("icate") + "ative" + "alize" +
                  "iciti" + "ical" + "ness" + "ful")

local step4a = eos(lpeg.P("ement") + "ance" + "ence" + "able" + "ible" +
                   "ment" + "ant" + "ent" + "ous" + "ism" + "ate" + "iti" +
                   "ive" + "ize" + "al" + "er" + "ic" + "ou")

local step4b = eos(lpeg.P("sion") + "tion")

local step5a = eos(lpeg.P("e"))
local step5b = eos(lpeg.P("ll"))

-- Stemming function
function stem(w)
   if #w < 3 then return w end

   local firstch = w:sub(1,1)
   if firstch == "y" then
      w = "Y" .. w:sub(2)
   end

   -- Step 1a
   if lpeg.match(step1a1, w) then
      w = w:sub(1, -3)
   elseif lpeg.match(step1a2, w) then
      w = w:sub(1, -2)
   end

   -- Step 1b
   if lpeg.match(step1b1, w) then
      if lpeg.match(mgr0, w:sub(1, -4)) then
         w = w:sub(1, -2)
      end
   else
      local suffix = lpeg.match(step1b2, w)

      if suffix then
         local stem = w:sub(1, -#suffix - 1)
         if lpeg.match(s_v, stem) then
            w = stem
            if lpeg.match(step1b3, w) then
               w = w .."e"
            -- Don't use lpeg, it's slower with backreferences
            elseif w:find("([^aeiouylsz])%1$") then
               w = w:sub(1, -2)
            elseif lpeg.match(e_v, w) then
               w = w .. "e"
            end
         end
      end
   end

   -- Step 1c
   if lpeg.match(step1c, w) then
      local stem = w:sub(1, -2)
      if lpeg.match(s_v, stem) then
         w = stem .. "i"
      end
   end

   -- Step 2
   local suffix = lpeg.match(step2, w)

   if suffix then
      local stem = w:sub(1, -#suffix - 1)
      if lpeg.match(mgr0, stem) then
         w = stem .. step2map[suffix]
      end
   end

   -- Step 3
   suffix = lpeg.match(step3, w)

   if suffix then
      local stem = w:sub(1, -#suffix - 1)
      if lpeg.match(mgr0, stem) then
         w = stem .. step3map[suffix]
      end
   end

   -- Step 4
   suffix = lpeg.match(step4a, w)

   if suffix then
      local stem = w:sub(1, -#suffix - 1)
      if lpeg.match(mgr1, stem) then
         w = stem
      end
   elseif lpeg.match(step4b, w) then
      local stem = w:sub(1, -4)
      if lpeg.match(mgr1, stem) then
         w = stem
      end
   end

   -- Step 5
   if lpeg.match(step5a, w) then
      local stem = w:sub(1, -2)
      if lpeg.match(mgr1, stem) or (lpeg.match(meq1, stem) and not lpeg.match(e_v, stem)) then
         w = stem
      end
   end

   if lpeg.match(step5b, w) and lpeg.match(mgr1, w) then
      w = w:sub(1, -2)
   end

   if firstch == "y" then
      w = "y" .. w:sub(2)
   end

   return w
end

--[[
function file(filename)
   local fp = io.open(filename, "r")
   if not fp then error("error opening file `"..tostring(filename).."'") end
   for l in fp:lines() do
      print(stem(l))
   end
   fp:close()
end

local args = {...}
if #args > 0 then
   for _, filename in ipairs(args) do
      file(filename)
   end
end
--]]

