#Copied from Data science from Scratch by Joel Grus
class Table:
    def __init__(self, columns):
        self.columns = columns #list containing names of columns
        self.rows = [] #starts as an empty list, will be a list of rows, each row will be a dict

    def __repr__(self):
        """representation of the table: columns then rows"""
        return str(self.columns) + "\n" + "\n".join(map(str,self.rows))

    def insert(self, row_values):
        if len(row_values) != len(self.columns):
            raise TypeError("wrong number of elements")
        row_dict = dict(zip(self.columns, row_values)) #each row is a dict from colun names to values
        self.rows.append(row_dict)
    
    def update(self, updates, predicate):
        """Updates existing data, two arguments:
        updates - a dict, keys are the columns that will be updated, values are the new values
        predicate - a predicate that returns True for rows that are to be updated"""
        for row in self.rows:
            if predicate(row): #checks if row is to be updated
                for column, new_value in updates.iteritems(): #
                    row[column] = new_value #replaces appropriate fields with the new values
    
    def delete(self, predicate=lambda row: True):
        """deletes all rows matching predicate
        if there is no predicate, all rows are deleted"""
        
