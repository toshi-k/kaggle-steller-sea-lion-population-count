
------------------------------
-- function
------------------------------

function load_mismatch_list(path)

	mismatch_list = {}

	fp = io.open(path, 'r')

	-- remove header
	str = fp:read()

	while str ~= nil do
		str = fp:read()
		-- print(str)
		if str ~= nil then
			table.insert(mismatch_list, str)
		end
	end
	fp:close()

	print('\tnum mismatch: ' .. #mismatch_list)
	print('\t\t' .. tostring(tablex.sub(mismatch_list,1,10)) .. '...'
	.. tostring(tablex.sub(mismatch_list,-10,Inf)))

	return mismatch_list

end
