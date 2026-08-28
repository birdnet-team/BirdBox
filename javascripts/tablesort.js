document$.subscribe(function() {
  var tables = document.querySelectorAll("article table:not([class])")
  tables.forEach(function(table) {
    
    // Find any cell marked as 'none' and apply it to the whole row
    table.querySelectorAll("td[data-sort-method='none']").forEach(function(td) {
      td.closest("tr").setAttribute("data-sort-method", "none");
    });

    new Tablesort(table)
  })
})