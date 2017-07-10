
------------------------------
-- functions
------------------------------

function setdiff(a_list, b_list)

	local match_list = {}
	for _, train in ipairs(a_list) do
		local is_mismatch = false
		for _, mismatch in ipairs(b_list) do
			if train == mismatch then
				is_mismatch = true
				break
			end
		end

		if not is_mismatch then
			table.insert(match_list, train)
		end
	end

	table.sort(match_list, function(a,b) return (tonumber(a) < tonumber(b)) end)

	print('\tnum match: ' .. #match_list)
	print('\t\t' .. tostring(tablex.sub(match_list,1,10)) .. '...'
			.. tostring(tablex.sub(match_list,-10,Inf)))

	return match_list
end
