import React from 'react';

const getIntensityClass = (value) => {
  switch (value) {
    case 1:
      return 'bg-slate-100 text-slate-600';
    case 2:
      return 'bg-sky-100 text-sky-700';
    case 3:
      return 'bg-blue-200 text-blue-800';
    case 4:
      return 'bg-orange-200 text-orange-800';
    case 5:
      return 'bg-red-500 text-white';
    default:
      return 'bg-slate-100 text-slate-600';
  }
};

export default function SeasonalHeatmap({ data }) {
  return (
    <div className="w-full overflow-x-auto">
      <div className="min-w-212.5">

        {/* Month headers */}
        <div className="grid grid-cols-[220px_repeat(12,minmax(45px,1fr))] gap-1">
          
          <div />

          {data.months.map((month) => (
            <div
              key={month}
              className="flex items-center justify-center py-2 text-xs font-medium text-slate-500"
            >
              {month}
            </div>
          ))}

          {/* Rows */}
          {data.rows.map((row) => (
            <React.Fragment key={row.label}>

              <div className="flex flex-col justify-center pr-3 py-2">
                <p className="text-sm font-medium text-slate-700">
                  {row.label}
                </p>

                <p className="mt-1 text-xs leading-relaxed text-slate-400">
                  {row.description}
                </p>
              </div>

              {row.values.map((value, index) => (
                <div
                  key={`${row.label}-${index}`}
                  title={`${row.label} — ${data.months[index]}: intensity ${value}`}
                  className={`flex h-12 items-center justify-center rounded-md text-sm font-semibold transition-transform hover:scale-105 ${getIntensityClass(value)}`}
                >
                  {value}
                </div>
              ))}

            </React.Fragment>
          ))}

        </div>

        {/* Legend */}
        <div className="mt-5 flex items-center justify-end gap-2">
          <span className="text-xs text-slate-500">
            Lower
          </span>

          {[1, 2, 3, 4, 5].map((value) => (
            <div
              key={value}
              className={`flex h-6 w-6 items-center justify-center rounded text-[10px] font-semibold ${getIntensityClass(value)}`}
            >
              {value}
            </div>
          ))}

          <span className="text-xs text-slate-500">
            Higher
          </span>
        </div>

      </div>
    </div>
  );
}